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
:class:`ValidationDataset` serves full images cropped to a multiple of
``scale``, uncached (each is decoded once per epoch, no repetition).

LR is derived via MATLAB-compatible antialiased bicubic resizing (see
:mod:`sisr.imresize`), comparable to published paper numbers.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..cache import LMDBCache, LMDBCacheBuildContext
from ..imresize import resize
from .base import SRDataset
from .hr_cache import CACHE_NAME, FORMAT_TAG, HEADER, compute_checksum, estimate_map_size
from .hr_cache import process_hr_image as _process_hr_image


class TrainDataset(SRDataset):
    """Random-crop HR/LR pairs for SRResNet-style training, HR held in an LMDB cache.

    On first instantiation with a given set of parameters, every HR image is
    decoded once and stored whole (uint8, RGB, with its own ``(H, W)`` header)
    in an LMDB database — see module docstring for why. Each ``__getitem__``
    then reads that image's cached bytes via a zero-copy
    :meth:`~sisr.cache.LMDBCache.get_buffer` view, takes a fresh random
    ``hr_crop_size`` square crop (plain numpy slicing — crop randomness must
    survive the cache, so only the *decode* is memoized, never the crop), and
    bicubic-downsamples it by ``scale`` (via :func:`sisr.imresize.resize`) to
    form the LR input. Unlike
    :class:`sisr.datasets.srcnn.TrainDataset` there is **no
    blur+downsample+upsample round-trip** and the LR is *not* upsampled back —
    the model is responsible for the ×``scale`` upsampling, so the LR tensor is
    ``hr_crop_size // scale`` on a side.

    Because the cache holds whole decoded images rather than crops, it is independent of
    ``hr_crop_size``/``crops_per_image``/``scale`` — the checksum keys only on
    the file manifest, so changing the crop recipe never invalidates the
    cache. HR is always served as RGB; Y/YCbCr selection happens downstream in
    :class:`SRLightning`.

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
        super().__init__()

        if hr_crop_size % scale != 0:
            raise ValueError(f"hr_crop_size ({hr_crop_size}) must be divisible by scale ({scale}).")

        self._index_images(img_dir)
        self.scale = scale
        self.hr_crop_size = hr_crop_size
        self.crops_per_image = crops_per_image
        self.build_num_workers = build_num_workers
        self.lr_size = hr_crop_size // scale

        cache_dir = Path(cache_dir) if cache_dir else self.img_dir / ".lmdb_cache"
        checksum = self._compute_checksum()

        self._cache = LMDBCache(
            cache_dir=cache_dir,
            name=CACHE_NAME,
            checksum=checksum,
            length=len(self.img_paths),
            map_size=estimate_map_size(self._collect_sizes()),
            metadata={"format": FORMAT_TAG},
            build_fn=self._build,
            use_tqdm=use_tqdm,
        )

    def _compute_checksum(self) -> str:
        """Computes a SHA-256 checksum over the file manifest only.

        No crop/scale parameter enters this hash: the cache stores whole raw
        images, so it is valid for any ``hr_crop_size``/``crops_per_image``/
        ``scale`` combination over the same file set. Delegates to
        :func:`~sisr.datasets.hr_cache.compute_checksum`, shared verbatim with
        :mod:`sisr.datasets.srcnn` so the same file set hashes identically
        regardless of which architecture asks first.

        Returns:
            A hex-encoded SHA-256 digest string.
        """
        return compute_checksum(self.img_paths)

    def _collect_sizes(self) -> list[tuple[int, int]]:
        """Reads each image's ``(h, w)`` from its file header, without decoding pixels.

        Returns:
            Each source image's ``(height, width)``, in :attr:`img_paths` order.
        """
        sizes = []
        for path in self.img_paths:
            img = Image.open(path)
            w, h = img.size
            img.close()
            sizes.append((h, w))
        return sizes

    def _build(self, ctx: LMDBCacheBuildContext) -> None:
        """Populates the LMDB cache by decoding each HR image once in parallel.

        Args:
            ctx (LMDBCacheBuildContext): Build context provided by :class:`~sisr.cache.LMDBCache`.
        """
        ctx.parallel_build(
            items=self.img_paths,
            process_fn=_process_hr_image,
            process_args=[(i,) for i in range(len(self.img_paths))],
            num_workers=self.build_num_workers,
            desc="Building SRResNet HR cache",
        )

    def __len__(self) -> int:
        return len(self.img_paths) * self.crops_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a ``(lr_tensor, hr_tensor)`` pair where ``hr_tensor`` is a random crop.

        ``hr_tensor`` is a random ``hr_crop_size`` square crop and
        ``lr_tensor`` is its bicubic downscale by ``scale`` (side
        ``hr_crop_size // scale``). Both are ``float32`` in ``[0, 1]``
        with shape ``(3, H, W)``.
        """
        img_idx = idx % len(self.img_paths)
        key = f"hr_{img_idx:08d}"

        with self._cache.get_buffer(key) as buf:
            if buf is None:
                raise KeyError(key)
            h, w = HEADER.unpack_from(buf, 0)
            if w < self.hr_crop_size or h < self.hr_crop_size:
                raise ValueError(
                    f"Image {self.img_paths[img_idx].name} ({w}x{h}) is smaller than "
                    f"hr_crop_size {self.hr_crop_size}."
                )
            # Read-only view into the LMDB transaction's mmap page — sliced and
            # copied out below, before the `with` block ends and the view dies.
            arr = np.frombuffer(buf, dtype=np.uint8, offset=HEADER.size).reshape(h, w, 3)
            top = random.randint(0, h - self.hr_crop_size)
            left = random.randint(0, w - self.hr_crop_size)
            hr_arr = arr[top : top + self.hr_crop_size, left : left + self.hr_crop_size, :].copy()

        lr_arr = resize(hr_arr, (self.lr_size, self.lr_size))
        lr_tensor = self._to_tensor(lr_arr)
        hr_tensor = self._to_tensor(hr_arr)

        return lr_tensor, hr_tensor


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
