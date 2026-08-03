"""SRCNN-style data pipeline.

LR is generated from HR via bicubic-down → bicubic-up so the LR patches share
the spatial size of the HR patches (pre-upsampled SRCNN formulation).
:class:`TrainDataset` caches whole decoded HR images (raw, uint8, headered)
through :mod:`~sisr.datasets.hr_cache`, shared verbatim with
:mod:`sisr.datasets.srresnet` — the same image directory produces exactly one
cache regardless of which architecture builds it first. Decoding a source
image once and slicing its deterministic sub-images out at load time is far
cheaper than re-decoding, and it decouples the cache from every
LR-generation parameter (``scale``) as well as the sub-image grid itself
(``subimg_size``, ``stride``): none of them affect the raw bytes stored, only
what gets sliced/degraded from them at read time.
:class:`ValidationDataset` generates LR pairs on the fly for full images.

LR degradation uses MATLAB-compatible antialiased bicubic resizing (see
:mod:`sisr.imresize`): its kernel widening on downscale is itself the
low-pass filter, so no separate blur step is needed.
"""

import bisect
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from ..cache import LMDBCacheBuildContext
from ..imresize import resize
from .base import HRCachedTrainDataset, SRDataset
from .hr_cache import process_hr_image as _process_hr_image


def _grid_dims(
    height: int, width: int, scale: int, sub_img_size: int, stride: int
) -> tuple[int, int]:
    """Returns the ``(n_rows, n_cols)`` sliding-window sub-image grid for one image.

    Single source of truth for the deterministic patch grid: :func:`_iter_patch_origins`
    and :meth:`TrainDataset.__getitem__`'s O(1) index -> origin lookup both derive
    positions from this, so they can never silently disagree and misalign an index.
    """
    h_crop = height - (height % scale)
    w_crop = width - (width % scale)
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
    formulation). Shared by :class:`TrainDataset` (per sub-image, at read
    time) and :class:`ValidationDataset` (per full image) so the degradation
    recipe cannot drift between the two.
    """
    h, w = hr_arr.shape[:2]
    down = resize(hr_arr, (h // scale, w // scale))
    return resize(down, (h, w))


class TrainDataset(HRCachedTrainDataset):
    """Dataset serving deterministic LR/HR sub-image pairs, HR held in an LMDB cache.

    On first instantiation with a given set of files, every HR image is
    decoded once and stored whole (uint8, RGB) in an LMDB database — see
    :mod:`~sisr.datasets.base`/:mod:`~sisr.datasets.hr_cache` for why. Each
    ``__getitem__`` maps its flat index to a source image and a deterministic
    ``(top, left)`` grid position in O(1) (via :func:`_grid_dims`), reads that
    image's cached bytes through :meth:`~sisr.datasets.base.HRCachedTrainDataset._read_hr`,
    slices out the ``subimg_size`` square HR sub-image, and degrades it to LR
    with :func:`_degrade`. The sliding-window grid itself — which sub-images
    exist and in what order — is derived from exactly one place
    (:func:`_grid_dims`), so indices can never silently misalign.

    A SHA-256 checksum over the file manifest (plus a format tag) is stored
    inside the LMDB so subsequent runs over the same files skip the build
    entirely — the checksum does not depend on
    ``subimg_size``/``stride``/``scale``, since none of them affect the raw
    bytes written.

    Reference:
        Image Super-Resolution Using Deep Convolutional Networks
        https://arxiv.org/pdf/1501.00092

    Args:
        img_dir: Directory of HR images.
        subimg_size: Spatial size of the square sub-images to extract.
        stride: Step size of the sliding window used for sub-image extraction.
        scale: Downscaling factor for generating LR sub-images.
        use_tqdm: Whether to display a progress bar during the LMDB build.
        cache_dir: Defaults to ``img_dir / '.lmdb_cache'``.
        build_num_workers: ``None`` (default) uses ``min(os.cpu_count() or
            1, num_images)``; ``<= 1`` effective workers runs an inline,
            no-subprocess build. Only affects cache construction, not data
            loading.

    Raises:
        ValueError: If no image files are found in ``img_dir``.
    """

    def __init__(
        self,
        img_dir: str | Path,
        subimg_size: int,
        stride: int,
        scale: int,
        use_tqdm: bool = False,
        cache_dir: str | Path | None = None,
        build_num_workers: int | None = None,
    ):
        self.sub_img_size = subimg_size
        self.stride = stride
        self.scale = scale
        super().__init__(
            img_dir, use_tqdm=use_tqdm, cache_dir=cache_dir, build_num_workers=build_num_workers
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
            n_rows, n_cols = _grid_dims(h, w, self.scale, self.sub_img_size, self.stride)
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
        :attr:`_img_offsets`/:attr:`_img_n_cols`, slices the HR sub-image out
        of that image's cached raw bytes (via
        :meth:`~sisr.datasets.base.HRCachedTrainDataset._read_hr`), and
        derives LR from it with :func:`_degrade` — numerically identical to
        computing both at build time, just performed at load time instead.

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

        lr_subimg = _degrade(hr_subimg, self.scale)

        return self._to_tensor(lr_subimg), self._to_tensor(hr_subimg)


class ValidationDataset(SRDataset):
    """Dataset that serves full-image LR/HR pairs for validation.

    Unlike :class:`TrainDataset` this dataset does not extract sub-images.
    Each item is a full image pair where the low-resolution version is
    produced by bicubic-downsampling and bicubic-upsampling back to the
    original size.

    Args:
        img_dir (str | Path): Directory containing the high-resolution
            images.
        scale (int): Downscaling factor for generating low-resolution images.

    Raises:
        ValueError: If no image files are found in ``img_dir``.
    """

    def __init__(self, img_dir: str | Path, scale: int):
        super().__init__()

        self._index_images(img_dir)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the LR/HR image pair at the given index.

        Args:
            idx (int): Zero-based image index.

        Returns:
            A ``(lr_tensor, hr_tensor)`` tuple of ``float32`` tensors
            with shape ``(C, H, W)`` and values in ``[0, 1]``.
        """
        path = self.img_paths[idx]
        arr = self._load_rgb(path)  # HWC uint8 RGB
        lr_arr = _degrade(arr, self.scale)

        return self._to_tensor(lr_arr), self._to_tensor(arr)
