"""SRCNN-style data pipeline.

LR is generated from HR via bicubic-down → bicubic-up so the LR patches share
the spatial size of the HR patches (pre-upsampled SRCNN formulation).
:class:`TrainDataset` caches whole decoded HR images (raw, uint8, headered)
through :mod:`~sisr.datasets.hr_cache`, shared verbatim with
:mod:`sisr.datasets.native_lr` — the same image directory produces exactly one
cache regardless of which architecture builds it first. Decoding a source
image once and slicing its deterministic sub-images out at load time is far
cheaper than re-decoding, and it decouples that cache from the sub-image
grid (``subimg_size``, ``stride``): neither affects the raw bytes stored.

**The whole image is downscaled once and cached** — the order the authors'
released ``demo_SR.m`` uses: ``modcrop`` then ``imresize`` down → up, and only
then patch extraction. Degrading an already-extracted patch instead resolves the
bicubic kernel's edge taps against the patch border: at x3 the interior is
bit-identical and the entire difference is a 4px border, 43% of a 33x33 patch.
That order is mandatory, not just cached this way for convenience — it is the
only way to get correct pixel values at a patch border, where the bicubic
kernel reaches outside the patch. :class:`TrainDataset` caches the LR-sized
downscale (``1/scale**2`` of HR pixels) and upsamples each training patch back
to HR size **at load time** with :func:`~sisr.utils.imresize.resize_window`,
which computes only the patch's window of the whole-image upsample and is
byte-identical to upsampling the whole image first and slicing it. The cached
working set is roughly HR bytes x ``(1 + 1/scale**2)`` instead of ``2x`` HR
bytes; the price is one small ``resize_window`` call per item instead of a
slice-copy. ``resize_window`` supports odd and power-of-two scales, so those are
the scales :class:`TrainDataset` accepts.

:class:`ValidationDataset` generates LR pairs on the fly for full images.

LR degradation uses MATLAB-compatible antialiased bicubic resizing (see
:mod:`sisr.utils.imresize`): its kernel widening on downscale is itself the
low-pass filter, so no separate blur step is needed.
"""

import bisect
import math
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

from ..utils.cache import LMDBCacheBuildContext
from ..utils.imresize import resize, resize_float, resize_window
from .base import HRCachedTrainDataset, SRDataset, as_scales
from .derived_cache import modcrop as _modcrop
from .derived_cache import modcrop_extent as _modcrop_extent
from .hr_cache import process_hr_image as _process_hr_image


def _grid_dims(
    height: int, width: int, scale: int, sub_img_size: int, stride: int
) -> tuple[int, int]:
    """Returns the ``(n_rows, n_cols)`` sliding-window sub-image grid for one image.

    Single source of truth for the deterministic patch grid: :func:`_iter_patch_origins`
    and :meth:`TrainDataset.__getitem__`'s O(1) index -> origin lookup both derive
    positions from this, so they can never silently disagree and misalign an index.

    The extent comes from :func:`_modcrop_extent`, the same function the
    validation loader crops by, so the two cannot drift apart again.
    """
    h_crop = _modcrop_extent(height, scale)
    w_crop = _modcrop_extent(width, scale)
    n_rows = len(range(0, h_crop - sub_img_size + 1, stride))
    n_cols = len(range(0, w_crop - sub_img_size + 1, stride))
    return n_rows, n_cols


def _iter_patch_origins(
    height: int,
    width: int,
    scale: int,
    sub_img_size: int,
    stride: int,
) -> Iterator[tuple[int, int]]:
    """Yields the ``(top, left)`` origin of every sliding-window sub-image, row-major.

    Built on :func:`_grid_dims` so this enumeration and
    :meth:`TrainDataset.__getitem__`'s O(1) index -> origin math can never
    disagree about ordering or count.
    """
    n_rows, n_cols = _grid_dims(height, width, scale, sub_img_size, stride)
    for row in range(n_rows):
        for col in range(n_cols):
            yield row * stride, col * stride


def _degrade(hr_arr: np.ndarray, scale: int) -> np.ndarray:
    """Derives an LR array from an HR array: bicubic-down → bicubic-up.

    Restores *hr_arr*'s original spatial size (SRCNN's pre-upsampled
    formulation). :class:`ValidationDataset` applies this per full image.
    :class:`TrainDataset` applies the downscale (down part) over the whole
    modcropped image and caches the result, then upsamples each sub-image
    patch at load time via :func:`~sisr.utils.imresize.resize_window`.
    """
    h, w = hr_arr.shape[:2]
    down = resize(hr_arr, (h // scale, w // scale))
    return resize(down, (h, w))


class TrainDataset(HRCachedTrainDataset):
    """Dataset serving deterministic LR/HR sub-image pairs, HR held in an LMDB cache.

    Every HR image is decoded once into an LMDB cache (uint8 RGB, whole) — see
    :mod:`~sisr.datasets.base` for why. Each ``__getitem__`` maps its flat index
    to a source image and a deterministic ``(top, left)`` grid position in O(1),
    and slices the ``subimg_size`` square directly out of the cached raw HR image
    at that position. The LR sub-image is upsampled from the cached whole-image
    LR downscale (1/scale**2 of HR) back to HR size **at load time** via
    :func:`~sisr.utils.imresize.resize_window` — byte-identical to upsampling
    the whole image first and slicing the same window, without materializing
    the full HR-sized plane.
    **The grid is derived from exactly one place** (:func:`_grid_dims`), so
    indices can never silently misalign.

    Because the downscale (HR→LR) happens over the whole image before extraction
    (see module docstring), ``subimg_size`` and ``scale`` are no longer coupled:
    there is no per-patch sampling grid left to shift, so ``33 % scale`` stops
    mattering.

    **The checksum keys only on the file manifest** (plus a format tag), not on
    ``subimg_size``/``stride``/``scale`` — none of those affect the bytes
    written — so subsequent runs over the same files skip the build.

    Reference:
        Image Super-Resolution Using Deep Convolutional Networks
        https://arxiv.org/pdf/1501.00092

    Args:
        img_dir: Directory of HR images.
        subimg_size: Spatial size of the square sub-images to extract.
        stride: Step size of the sliding window used for sub-image extraction.
        scale: Downscaling factor for generating LR sub-images, or a list of
            them for multi-scale training: **a scale is drawn per patch**, so a
            batch mixes scales (fine here, every patch is on the HR grid and the
            model never sees the scale). The patch grid then comes from the
            modcrop of the scales' least common multiple, which is inside every
            scale's own modcrop, so each patch is valid at each scale. One derived
            LR cache is built per scale.
        use_tqdm: Whether to display a progress bar during the LMDB build.
        cache_dir: Defaults to ``img_dir / '.lmdb_cache'``.
        build_num_workers: ``None`` (default) uses ``min(os.cpu_count() or
            1, num_images)``; ``<= 1`` effective workers runs an inline,
            no-subprocess build. Only affects cache construction, not data
            loading.

    Raises:
        ValueError: If no image files are found in ``img_dir``, or *scale* is
            neither odd nor a power of two.
    """

    def __init__(
        self,
        img_dir: str | Path,
        subimg_size: int,
        stride: int,
        scale: int | Sequence[int],
        use_tqdm: bool = False,
        cache_dir: str | Path | None = None,
        build_num_workers: int | None = None,
    ):
        self.scales = as_scales(scale)
        # Fail here, before the cache build, rather than in the first loader worker.
        for s in self.scales:
            resize_window(np.zeros((1, 1), dtype=np.uint8), (s, s), 0, 1, 0, 1)
        self.sub_img_size = subimg_size
        self.stride = stride
        self.scale = self.scales[0] if len(self.scales) == 1 else list(self.scales)
        self.scale_per_batch = False
        self._grid_scale = math.lcm(*self.scales)
        super().__init__(
            img_dir,
            scale=self.scales,
            derived_kind="lr",
            use_tqdm=use_tqdm,
            cache_dir=cache_dir,
            build_num_workers=build_num_workers,
        )
        self._img_offsets, self._img_n_cols, self._total_patches = self._compute_grid()

    def _compute_grid(self) -> tuple[list[int], list[int], int]:
        """Computes the deterministic sub-image grid from :attr:`_img_sizes`.

        Single source of truth for the patch grid: :meth:`__len__` and
        :meth:`__getitem__`'s O(1) index -> ``(top, left)`` lookup both derive
        from the values this returns, so they can never disagree.

        Returns:
            ``(offsets, n_cols, total)`` where *offsets* is the cumulative
            starting patch index per image, *n_cols* is each image's column
            count (needed to invert a flat patch index back to a grid
            position), and *total* is the grand total of patches.
        """
        offsets: list[int] = []
        n_cols_list: list[int] = []
        offset = 0
        for h, w in self._img_sizes:
            n_rows, n_cols = _grid_dims(h, w, self._grid_scale, self.sub_img_size, self.stride)
            n_cols_list.append(n_cols)
            offsets.append(offset)
            offset += n_rows * n_cols
        return offsets, n_cols_list, offset

    def _build(self, ctx: LMDBCacheBuildContext) -> None:
        """Populates the LMDB cache by decoding each HR image once, in parallel."""
        self._parallel_build_hr(ctx, _process_hr_image)

    def __len__(self) -> int:
        return self._total_patches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the LR/HR sub-image pair at the given index.

        Locates *idx*'s source image and grid position in O(1) via
        :attr:`_img_offsets`/:attr:`_img_n_cols`, slices the HR sub-image
        directly out of that image's cached raw bytes (via
        :meth:`~sisr.datasets.base.HRCachedTrainDataset._read_hr`), and
        upsamples the LR sub-image from the cached whole-image LR downscale
        (via :meth:`~sisr.datasets.base.HRCachedTrainDataset._read_derived`
        and :func:`~sisr.utils.imresize.resize_window`). The downscale and
        upsample slicing are aligned to the same grid position, so both results
        correspond to the correct spatial patch.

        Args:
            idx (int): Zero-based sub-image index.

        Returns:
            A ``(lr_tensor, hr_tensor)`` tuple of ``float32`` tensors
            with shape ``(C, H, W)`` and values in ``[0, 1]``.

        Raises:
            IndexError: If *idx* is outside ``[0, len(self))``.
            KeyError: If the backing LMDB entry for the source image is
                missing (a corrupt/incomplete cache).
        """
        if not 0 <= idx < self._total_patches:
            raise IndexError(idx)

        img_idx = bisect.bisect_right(self._img_offsets, idx) - 1
        local_idx = idx - self._img_offsets[img_idx]
        row, col = divmod(local_idx, self._img_n_cols[img_idx])
        top, left = row * self.stride, col * self.stride

        with self._read_hr(img_idx) as arr:
            hr_subimg = arr[
                top : top + self.sub_img_size, left : left + self.sub_img_size, :
            ].copy()

        # No draw for one scale: the RNG stream a single-scale run consumes is unchanged.
        scale = random.choice(self.scales) if len(self.scales) > 1 else self.scales[0]
        h_full, w_full = self._img_sizes[img_idx]
        h_crop = _modcrop_extent(h_full, scale)
        w_crop = _modcrop_extent(w_full, scale)
        with self._read_derived(img_idx, scale) as plane:
            lr_subimg = resize_window(
                plane,
                (h_crop, w_crop),
                top,
                self.sub_img_size,
                left,
                self.sub_img_size,
            )

        return self._to_tensor(lr_subimg), self._to_tensor(hr_subimg)


class ValidationDataset(SRDataset):
    """Dataset that serves full-image LR/HR pairs for validation.

    Unlike :class:`TrainDataset` this dataset does not extract sub-images.
    Each item is a full image pair where the low-resolution version is
    produced by bicubic-downsampling and bicubic-upsampling back to the
    original size.

    **The HR image is modcropped first** (:func:`_modcrop`), matching the
    authors' released order and what :class:`TrainDataset`'s grid already did.
    So the HR served here is up to ``scale - 1`` pixels smaller per axis than
    the file on disk -- that is the reference the paper's numbers are computed
    against.

    Args:
        img_dir (str | Path): Directory containing the high-resolution
            images.
        scale (int): Downscaling factor for generating low-resolution images.
        float_degrade (bool): Degrade in floating point -- no uint8 re-quantization
            between the down and up passes and no final rounding or clamp
            (:func:`~sisr.utils.imresize.resize_float`) -- as ``VDSR.m`` and
            ``demo_SR.m`` do in ``single`` precision. The served LR is then not an 8-bit
            image and may leave ``[0, 1]`` by a bicubic overshoot. Evaluation only: the
            training dataset has no such option and the uint8 path is untouched.
            Default ``False``.

    Raises:
        ValueError: If no image files are found in ``img_dir``.
    """

    def __init__(self, img_dir: str | Path, scale: int, float_degrade: bool = False):
        super().__init__()

        self._index_images(img_dir)
        self.scale = scale
        self.float_degrade = float_degrade

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the LR/HR image pair at the given index.

        Args:
            idx (int): Zero-based image index.

        Returns:
            A ``(lr_tensor, hr_tensor)`` tuple of ``float32`` tensors
            with shape ``(C, H, W)`` and values in ``[0, 1]``. Both are
            modcropped to a whole multiple of ``scale``, so ``H``/``W`` may be
            up to ``scale - 1`` smaller than the source file's.
        """
        path = self.img_paths[idx]
        arr = _modcrop(self._load_rgb(path), self.scale)  # HWC uint8 RGB
        if self.float_degrade:
            h, w = arr.shape[:2]
            down = resize_float(arr.astype(np.float64), (h // self.scale, w // self.scale))
            lr = torch.from_numpy((resize_float(down, (h, w)) / 255.0).astype(np.float32))
            return lr.permute(2, 0, 1).contiguous(), self._to_tensor(arr)
        lr_arr = _degrade(arr, self.scale)

        return self._to_tensor(lr_arr), self._to_tensor(arr)
