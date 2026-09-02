"""SRResNet-style data pipeline.

LR is the bicubic downscale of HR by ``scale`` (no upsample round-trip);
the model is responsible for the ×``scale`` upsampling. :class:`TrainDataset`
caches full decoded HR images (raw, uint8, headered) through
:mod:`~sisr.datasets.hr_cache`, shared verbatim with :mod:`sisr.datasets.srcnn`
— the same image directory produces exactly one cache regardless of which
architecture builds it first. Decoding a DIV2K PNG costs ~109ms, while the
96x96 crop kept from it costs ~1ms, so re-decoding per crop is ~99% wasted
work. The random crop itself is **not** cached (that would defeat crop
randomness); it is drawn fresh, from the cached raw array, on every
``__getitem__`` call.

**The whole image is downscaled once, and the pair is cropped from HR and LR
together** — the order BasicSR, EDSR and DIV2K's own distributed LR archives
all use. Downscaling an extracted crop instead resolves the bicubic kernel's
edge taps against the crop border: measured on DIV2K, every crop's interior
stays bit-identical while the 2px LR border — 30.6% of a 24x24 patch — differs
by up to 22/255. The whole-image plane is cached by
:mod:`~sisr.datasets.derived_cache`.
:class:`ValidationDataset` serves full images cropped to a multiple of
``scale``, uncached (each is decoded once per epoch, no repetition).

LR is derived via MATLAB-compatible antialiased bicubic resizing (see
:mod:`sisr.utils.imresize`), comparable to published paper numbers.
"""

import random
from pathlib import Path

import torch

from ..utils.cache import LMDBCacheBuildContext
from ..utils.imresize import resize
from .base import HRCachedTrainDataset, SRDataset
from .hr_cache import process_hr_image as _process_hr_image


class TrainDataset(HRCachedTrainDataset):
    """Random-crop HR/LR pairs for SRResNet-style training, HR held in an LMDB cache.

    Every HR image is decoded once into an LMDB cache (uint8 RGB, whole, with
    its ``(H, W)`` header) — see :mod:`~sisr.datasets.base` for why. Each
    ``__getitem__`` draws a fresh random position in **LR** coordinates and
    slices the pair out of the cached whole-image HR and LR planes at that
    position — the LR one already downscaled, so no kernel ever reaches past a
    crop edge. **Only the planes are memoized, never the crop** — crop
    randomness has to survive the cache.

    Drawing in LR coordinates makes every HR offset a multiple of ``scale``,
    exactly as EDSR does. That is 16x fewer distinct positions at x4 than
    unaligned offsets would give — 153,892 rather than 2,452,645 on a
    2040x1356 image — and still ~123M across an 800-image training set.

    Unlike :class:`sisr.datasets.srcnn.TrainDataset` there is **no
    downsample+upsample round-trip**: the LR is not upsampled back, the model
    owns the x``scale`` upsampling, and the LR tensor is
    ``hr_crop_size // scale`` a side.

    **The raw-HR checksum keys only on the file manifest**, so changing
    ``hr_crop_size``/``crops_per_image``/``scale`` never invalidates it. The
    derived LR plane is keyed separately and *does* carry ``scale``.
    HR is always RGB; Y/YCbCr selection happens downstream.

    Reference:
        Photo-Realistic Single Image Super-Resolution Using a Generative
        Adversarial Network (https://arxiv.org/pdf/1609.04802)

    Args:
        img_dir: Directory of HR images.
        scale: Upscaling factor. ``hr_crop_size`` must be divisible by it.
        hr_crop_size: Side length of the square HR crop.
        crops_per_image: Random crops drawn per image per epoch (the dataset
            length is ``len(images) * crops_per_image``). Each draw re-reads
            the same cached array (a cheap mmap access, not a decode) and
            takes an independently random crop. Defaults to ``1``.
        use_tqdm: Whether to display a progress bar during the LMDB build.
        cache_dir: Defaults to ``img_dir / '.lmdb_cache'``.
        build_num_workers: ``None`` (default) uses ``min(os.cpu_count() or
            1, num_images)``; ``<= 1`` effective workers runs an inline,
            no-subprocess build. Only affects cache construction, not data
            loading.

    Raises:
        ValueError: If no images are found, or ``hr_crop_size`` is not
            divisible by ``scale``.
    """

    def __init__(
        self,
        img_dir: str | Path,
        scale: int,
        hr_crop_size: int,
        crops_per_image: int = 1,
        use_tqdm: bool = False,
        cache_dir: str | Path | None = None,
        build_num_workers: int | None = None,
    ):
        if hr_crop_size % scale != 0:
            raise ValueError(f"hr_crop_size ({hr_crop_size}) must be divisible by scale ({scale}).")

        self.scale = scale
        self.hr_crop_size = hr_crop_size
        self.crops_per_image = crops_per_image
        self.lr_size = hr_crop_size // scale
        super().__init__(
            img_dir,
            scale=scale,
            derived_kind="lr",
            use_tqdm=use_tqdm,
            cache_dir=cache_dir,
            build_num_workers=build_num_workers,
        )

    def _build(self, ctx: LMDBCacheBuildContext) -> None:
        """Populates the LMDB cache by decoding each HR image once in parallel."""
        self._parallel_build_hr(ctx, _process_hr_image)

    def __len__(self) -> int:
        return len(self.img_paths) * self.crops_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a ``(lr_tensor, hr_tensor)`` pair cropped from the whole-image planes.

        A random position is drawn in LR coordinates; ``lr_tensor`` is the
        ``hr_crop_size // scale`` square there and ``hr_tensor`` the
        ``hr_crop_size`` square at ``scale`` times that position, so the two are
        aligned by construction. Both are ``float32`` in ``[0, 1]`` with shape
        ``(3, H, W)``.

        Raises:
            ValueError: If the image is smaller than ``hr_crop_size`` after the
                modcrop the LR plane was derived under.
        """
        img_idx = idx % len(self.img_paths)

        with self._read_derived(img_idx) as lr_plane:
            lr_h, lr_w = lr_plane.shape[:2]
            if lr_h < self.lr_size or lr_w < self.lr_size:
                raise ValueError(
                    f"Image {self.img_paths[img_idx].name} is {lr_w * self.scale}x"
                    f"{lr_h * self.scale} after modcrop, smaller than hr_crop_size "
                    f"{self.hr_crop_size}."
                )
            top = random.randint(0, lr_h - self.lr_size)
            left = random.randint(0, lr_w - self.lr_size)
            lr_arr = lr_plane[top : top + self.lr_size, left : left + self.lr_size, :].copy()

        hr_top, hr_left = top * self.scale, left * self.scale
        with self._read_hr(img_idx) as arr:
            hr_arr = arr[
                hr_top : hr_top + self.hr_crop_size,
                hr_left : hr_left + self.hr_crop_size,
                :,
            ].copy()

        return self._to_tensor(lr_arr), self._to_tensor(hr_arr)


class ValidationDataset(SRDataset):
    """Full-image HR with bicubic-downsampled LR for SRResNet validation/test.

    Each item is a full image pair. The HR image is cropped to a multiple of
    ``scale`` so the model's ×``scale`` output lands exactly on the HR size;
    the LR is the bicubic downscale by ``scale`` (no upsample round-trip).
    HR is always served as RGB.

    Args:
        img_dir (str | Path): Directory containing the high-resolution images.
        scale (int): Upscaling factor.

    Raises:
        ValueError: If no images are found in ``img_dir``.
    """

    def __init__(self, img_dir: str | Path, scale: int):
        super().__init__()

        self._index_images(img_dir)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a ``(lr_tensor, hr_tensor)`` pair for the image at ``idx``.

        ``hr_tensor`` is the HR image cropped to a multiple of ``scale``,
        and ``lr_tensor`` is its bicubic downscale by ``scale``. Both are
        ``float32`` in ``[0, 1]``.
        """
        path = self.img_paths[idx]
        arr = self._load_rgb(path)  # HWC uint8 RGB

        h, w = arr.shape[:2]
        h_crop = h - (h % self.scale)
        w_crop = w - (w % self.scale)
        hr_arr = arr[:h_crop, :w_crop, :]  # deterministic exact-corner crop

        lr_arr = resize(hr_arr, (h_crop // self.scale, w_crop // self.scale))

        lr_tensor = self._to_tensor(lr_arr)
        hr_tensor = self._to_tensor(hr_arr)

        return lr_tensor, hr_tensor
