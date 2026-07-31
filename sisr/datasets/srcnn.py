"""SRCNN-style data pipeline.

LR is generated from HR via (optional blur →) bicubic-down → bicubic-up so
the LR patches share the spatial size of the HR patches (pre-upsampled SRCNN
formulation). :class:`TrainDataset` caches whole decoded HR images (raw,
uint8) through :class:`~sisr.cache.LMDBCache`, mirroring
:mod:`sisr.datasets.srresnet` — decoding a source image once and slicing its
deterministic sub-images out at load time is far cheaper than re-decoding,
and it decouples the cache from every LR-generation parameter (``scale``,
``blur_sigma``, ``resize_backend``) as well as the sub-image grid itself
(``subimg_size``, ``stride``): none of them affect the raw bytes stored, only
what gets sliced/degraded from them at read time. :class:`ValidationDataset`
generates LR pairs on the fly for full images.

The resize/degradation backend is selectable (see :mod:`sisr.imresize`):
``'matlab'`` (default) is antialiased and needs no separate blur step;
``'cv2'`` has no antialiasing of its own, so ``blur_sigma`` stands in for it
and only has an effect on that path.
"""

import bisect
import hashlib
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..cache import LMDBCache, LMDBCacheBuildContext
from ..imresize import ResizeBackend, resize
from .base import SRDataset


def _check_blur_sigma(resize_backend: ResizeBackend, blur_sigma: float | None) -> float | None:
    """Validates the ``blur_sigma``/``resize_backend`` pairing at construction time.

    ``'matlab'``'s antialiasing kernel widening already acts as the low-pass
    filter; stacking an explicit Gaussian blur on top would double-blur and
    push PSNR away from published values, so setting ``blur_sigma`` with it
    is rejected outright rather than silently ignored. ``'cv2'`` has no
    antialiasing of its own, so it falls back to the historical default of
    ``1.0`` when unset.

    Args:
        resize_backend: ``'matlab'`` or ``'cv2'``.
        blur_sigma: The value passed in, or ``None``.

    Returns:
        ``None`` for ``'matlab'``; a concrete sigma for ``'cv2'``.

    Raises:
        ValueError: If ``resize_backend='matlab'`` and *blur_sigma* is set.
    """
    if resize_backend == "matlab":
        if blur_sigma is not None:
            raise ValueError(
                "blur_sigma is meaningless with resize_backend='matlab': its antialiasing "
                "kernel widening already is the low-pass filter, so an explicit Gaussian blur "
                "on top only double-blurs and pushes PSNR away from published values. Pass "
                "resize_backend='cv2' if you need blur_sigma, or omit blur_sigma for 'matlab'."
            )
        return None
    return 1.0 if blur_sigma is None else blur_sigma


def _grid_dims(
    height: int, width: int, scale: int, sub_img_size: int, stride: int
) -> tuple[int, int]:
    """Returns the ``(n_rows, n_cols)`` sliding-window sub-image grid for one image.

    Single source of truth for the deterministic patch grid: :func:`_iter_patch_origins`
    (the enumeration used by tests and the module docstring) and
    :meth:`TrainDataset.__getitem__` (an O(1) index -> origin lookup, since the
    cache no longer stores one entry per patch) both derive positions from
    this, so they can never silently disagree and misalign an index.

    Args:
        height (int): Full image height in pixels.
        width (int): Full image width in pixels.
        scale (int): Downscaling factor; each axis is cropped to a multiple of it.
        sub_img_size (int): Side length of the square sub-image window.
        stride (int): Step size of the sliding window.

    Returns:
        ``(n_rows, n_cols)`` — the number of valid window positions per axis.
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

    Args:
        height (int): Full image height in pixels.
        width (int): Full image width in pixels.
        scale (int): Downscaling factor; each axis is cropped to a multiple of it.
        sub_img_size (int): Side length of the square sub-image window.
        stride (int): Step size of the sliding window.

    Yields:
        ``(top, left)`` pixel offsets of each sub-image's top-left corner.
    """
    n_rows, n_cols = _grid_dims(height, width, scale, sub_img_size, stride)
    for row in range(n_rows):
        for col in range(n_cols):
            yield row * stride, col * stride


def _degrade(
    hr_arr: np.ndarray,
    scale: int,
    kernel: int | None,
    blur_sigma: float | None,
    resize_backend: ResizeBackend,
) -> np.ndarray:
    """Derives an LR array from an HR array: (optional blur →) bicubic-down → bicubic-up.

    Restores *hr_arr*'s original spatial size (SRCNN's pre-upsampled
    formulation). Shared by :class:`TrainDataset` (per sub-image, at read
    time) and :class:`ValidationDataset` (per full image) so the degradation
    recipe cannot drift between the two.

    Args:
        hr_arr (np.ndarray): ``(H, W, 3)`` uint8 RGB array.
        scale (int): Downscaling factor.
        kernel (int | None): Odd Gaussian kernel size, or ``None`` on ``'matlab'``.
        blur_sigma (float | None): Gaussian sigma paired with *kernel*; ignored if
            *kernel* is ``None``.
        resize_backend (ResizeBackend): ``'matlab'`` or ``'cv2'``.

    Returns:
        The degraded array, same ``(H, W, 3)`` shape as *hr_arr*.
    """
    h, w = hr_arr.shape[:2]
    to_degrade = hr_arr
    if kernel is not None:
        to_degrade = cv2.GaussianBlur(hr_arr, (kernel, kernel), sigmaX=blur_sigma)
    down = resize(to_degrade, (h // scale, w // scale), backend=resize_backend)
    return resize(down, (h, w), backend=resize_backend)


def _process_hr_image(path: Path, idx: int) -> list[tuple[str, bytes]]:
    """Decodes one HR image and returns its single LMDB entry.

    Top-level (not a method) so it can be pickled by ``ProcessPoolExecutor``
    across all platforms, mirroring
    :func:`sisr.datasets.srresnet._process_hr_image`. Sub-image extraction and
    LR derivation both moved to :meth:`TrainDataset.__getitem__`, so the build
    only has to decode once per image — independent of
    ``subimg_size``/``stride``/``scale``.

    Args:
        path (Path): File path of the high-resolution image.
        idx (int): This image's position in the manifest — determines its LMDB key.

    Returns:
        A single-element list ``[(f'hr_{idx:08d}', raw_bytes)]`` where
        *raw_bytes* is the ``(H, W, 3)`` uint8 RGB array's bytes.
    """
    arr = np.array(Image.open(path).convert("RGB"))
    return [(f"hr_{idx:08d}", arr.tobytes())]


class TrainDataset(SRDataset):
    """Dataset that serves deterministic LR/HR sub-image pairs, HR served from
    an LMDB cache of full raw images.

    On first instantiation with a given set of files, every HR image is
    decoded once and stored whole (uint8, RGB) in an LMDB database — see the
    module docstring for why. Each ``__getitem__`` maps its flat index to a
    source image and a deterministic ``(top, left)`` grid position in O(1)
    (via :func:`_grid_dims`), reads that image's cached bytes through a
    zero-copy :meth:`~sisr.cache.LMDBCache.get_buffer` view, slices out the
    ``subimg_size`` square HR sub-image, and degrades it to LR with
    :func:`_degrade`. The sliding-window grid itself — which sub-images exist
    and in what order — is unaffected by this change: it is still derived
    from exactly one place (:func:`_grid_dims`), so indices can never
    silently misalign.

    A SHA-256 checksum over the file manifest (plus a format tag) is stored
    inside the LMDB so subsequent runs over the same files skip the build
    entirely — the checksum no longer depends on
    ``subimg_size``/``stride``/``scale``/``blur_sigma``/``resize_backend``,
    since none of them affect the raw bytes written.

    Reference:
        Image Super-Resolution Using Deep Convolutional Networks
        https://arxiv.org/pdf/1501.00092

    Args:
        img_dir (str | Path): Directory containing the high-resolution
            images.
        subimg_size (int): Spatial size of the square sub-images to extract.
        stride (int): Step size of the sliding window used for sub-image
            extraction.
        scale (int): Downscaling factor for generating low-resolution
            sub-images.
        blur_sigma (float | None): Sigma of the Gaussian blur applied before
            downsampling. Only meaningful (and must be set) on the ``'cv2'``
            backend, which has no antialiasing of its own; falls back to the
            historical default of ``1.0`` if left ``None`` there. Must be
            ``None`` on ``'matlab'`` (the default) — its kernel widening
            already is the antialiasing low-pass, so an explicit blur on top
            only double-blurs; passing a value raises ``ValueError``.
        resize_backend (ResizeBackend): ``'matlab'`` (default) — MATLAB-
            compatible antialiased bicubic, comparable to published paper
            numbers. ``'cv2'`` — ``cv2.INTER_CUBIC``, no antialiasing;
            kept so LMDB caches built before this module existed stay
            reproducible. See :mod:`sisr.imresize`. Only the LR derivation
            is affected — the HR cache stores raw pixels regardless, so
            switching backends never invalidates it.
        use_tqdm (bool): Whether to display a progress bar during the LMDB
            build.  Defaults to ``False``.
        cache_dir (str | Path | None): Directory in which to store the
            LMDB cache.  Defaults to ``img_dir / '.lmdb_cache'``.
        build_num_workers (int | None): Number of worker processes for the
            one-time LMDB build.  ``None`` (default) uses
            ``min(os.cpu_count() or 1, num_images)``; a value that resolves to
            ``<= 1`` effective workers runs an inline, no-subprocess build.
            Only affects cache construction, not data loading.

    Raises:
        ValueError: If no image files are found in ``img_dir``, or if
            *blur_sigma* is set together with ``resize_backend='matlab'``.
    """

    def __init__(
        self,
        img_dir: str | Path,
        subimg_size: int,
        stride: int,
        scale: int,
        blur_sigma: float | None = None,
        resize_backend: ResizeBackend = "matlab",
        use_tqdm: bool = False,
        cache_dir: str | Path | None = None,
        build_num_workers: int | None = None,
    ):
        super().__init__()

        self._index_images(img_dir)
        self.sub_img_size = subimg_size
        self.stride = stride
        self.scale = scale
        self.blur_sigma = _check_blur_sigma(resize_backend, blur_sigma)
        self.resize_backend = resize_backend
        self.build_num_workers = build_num_workers
        self._kernel = (
            2 * math.ceil(3.0 * self.blur_sigma) + 1 if self.blur_sigma is not None else None
        )  # odd, covers +/-3 sigma

        cache_dir = Path(cache_dir) if cache_dir else self.img_dir / ".lmdb_cache"
        checksum = self._compute_checksum()
        self._img_sizes, self._img_offsets, self._img_n_cols, self._total_patches = (
            self._compute_grid()
        )

        # Exact sizes from image headers + 10% slack. If it is ever exceeded,
        # lmdb raises MapFullError mid-build and the checksum is never written,
        # so the partial DB reads as stale and is rebuilt rather than silently
        # serving truncated data. Same unhandled-overflow behaviour as SRResNet's cache.
        raw_bytes = sum(h * w * 3 for h, w in self._img_sizes)
        map_size = max(int(raw_bytes * 1.1), 64 * 1024 * 1024)

        self._cache = LMDBCache(
            cache_dir=cache_dir,
            name="srcnn_hr",
            checksum=checksum,
            length=len(self.img_paths),
            map_size=map_size,
            metadata={"format": "srcnn_hr_v1"},
            build_fn=self._build,
            use_tqdm=use_tqdm,
        )

    def _compute_checksum(self) -> str:
        """Computes a SHA-256 checksum over the file manifest only.

        The cache stores whole raw HR images, so — unlike the LR-patch cache
        this replaced — none of ``subimg_size``/``stride``/``scale``/
        ``blur_sigma``/``resize_backend`` enter the hash: they only affect
        what gets sliced and degraded at read time, never what is written to
        LMDB. The ``format=srcnn_hr_v1`` tag (paired with a new LMDB cache
        ``name``, see :meth:`__init__`) ensures a pre-HR-only cache is never
        misread under this format rather than rebuilt.

        Returns:
            A hex-encoded SHA-256 digest string.
        """
        file_manifest = ",".join(f"{p.name}:{p.stat().st_size}" for p in self.img_paths)
        canonical = "|".join([file_manifest, "format=srcnn_hr_v1"])
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _compute_grid(self) -> tuple[list[tuple[int, int]], list[int], list[int], int]:
        """Reads image dimensions (without decoding pixels) to compute the
        deterministic sub-image grid for every source image.

        Single source of truth for the patch grid: :meth:`__len__` and
        :meth:`__getitem__`'s O(1) index -> ``(top, left)`` lookup both derive
        from the values this returns, so they can never disagree (P2.5).

        Returns:
            ``(sizes, offsets, n_cols, total)`` where *sizes* is each image's
            ``(h, w)``, *offsets* is the cumulative starting patch index per
            image, *n_cols* is each image's column count (needed to invert a
            flat patch index back to a grid position), and *total* is the
            grand total of patches.
        """
        sizes: list[tuple[int, int]] = []
        offsets: list[int] = []
        n_cols_list: list[int] = []
        offset = 0
        for path in self.img_paths:
            img = Image.open(path)
            w, h = img.size
            img.close()
            n_rows, n_cols = _grid_dims(h, w, self.scale, self.sub_img_size, self.stride)
            sizes.append((h, w))
            n_cols_list.append(n_cols)
            offsets.append(offset)
            offset += n_rows * n_cols
        return sizes, offsets, n_cols_list, offset

    def _build(self, ctx: LMDBCacheBuildContext) -> None:
        """Populates the LMDB cache by decoding each HR image once, in parallel.

        Args:
            ctx (LMDBCacheBuildContext): Build context provided by
                :class:`LMDBCache`.
        """
        ctx.parallel_build(
            items=self.img_paths,
            process_fn=_process_hr_image,
            process_args=[(i,) for i in range(len(self.img_paths))],
            num_workers=self.build_num_workers,
            desc="Building LMDB cache",
        )

    def __len__(self) -> int:
        return self._total_patches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the LR/HR sub-image pair at the given index.

        Locates *idx*'s source image and grid position in O(1) via
        :attr:`_img_offsets`/:attr:`_img_n_cols`, slices the HR sub-image out
        of that image's cached raw bytes (a zero-copy
        :meth:`~sisr.cache.LMDBCache.get_buffer` view, copied out before the
        transaction closes), and derives LR from it with :func:`_degrade` —
        numerically identical to computing both at build time, just performed
        at load time instead.

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
        h, w = self._img_sizes[img_idx]

        key = f"hr_{img_idx:08d}"
        with self._cache.get_buffer(key) as buf:
            if buf is None:
                raise KeyError(key)
            # Read-only view into the LMDB transaction's mmap page — sliced and
            # copied out below, before the `with` block ends and the view dies.
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            hr_subimg = arr[
                top : top + self.sub_img_size, left : left + self.sub_img_size, :
            ].copy()

        lr_subimg = _degrade(
            hr_subimg, self.scale, self._kernel, self.blur_sigma, self.resize_backend
        )

        return self._to_tensor(lr_subimg), self._to_tensor(hr_subimg)


class ValidationDataset(SRDataset):
    """Dataset that serves full-image LR/HR pairs for validation.

    Unlike :class:`TrainDataset` this dataset does not extract sub-images.
    Each item is a full image pair where the low-resolution version is
    produced by (optionally, ``'cv2'``-only) Gaussian-blurring, then
    bicubic-downsampling and bicubic-upsampling back to the original size.

    Args:
        img_dir (str | Path): Directory containing the high-resolution
            images.
        scale (int): Downscaling factor for generating low-resolution images.
        blur_sigma (float | None): Sigma of the Gaussian blur applied before
            downsampling.  Must match :class:`TrainDataset` to keep train/val
            LR generation consistent. Same ``resize_backend`` pairing rules
            as :class:`TrainDataset` apply (``None``-only on ``'matlab'``,
            defaults to ``1.0`` on ``'cv2'``).
        resize_backend (ResizeBackend): ``'matlab'`` (default) or ``'cv2'``;
            see :class:`TrainDataset` / :mod:`sisr.imresize`.

    Raises:
        ValueError: If no image files are found in ``img_dir``, or if
            *blur_sigma* is set together with ``resize_backend='matlab'``.
    """

    def __init__(
        self,
        img_dir: str | Path,
        scale: int,
        blur_sigma: float | None = None,
        resize_backend: ResizeBackend = "matlab",
    ):
        super().__init__()

        self._index_images(img_dir)
        self.scale = scale
        self.blur_sigma = _check_blur_sigma(resize_backend, blur_sigma)
        self.resize_backend = resize_backend

        self._kernel = (
            2 * math.ceil(3.0 * self.blur_sigma) + 1 if self.blur_sigma is not None else None
        )  # odd, covers ±3σ

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
        lr_arr = _degrade(arr, self.scale, self._kernel, self.blur_sigma, self.resize_backend)

        return self._to_tensor(lr_arr), self._to_tensor(arr)
