"""Shared base for SR datasets — enforces the ``.img_paths`` contract.

All four architecture datasets (SRCNN / SRResNet train + validation) subclass
:class:`SRDataset`. It centralises the file discovery (extension-allowlisted
glob + empty-directory guard), the RGB image load, the ``uint8`` HWC →
``float32`` CHW ``[0, 1]`` tensor adapter, and declares the ``.img_paths``
filename contract that :class:`~sisr.training.SRDataModule` and
:class:`~sisr.training.callbacks.BenchmarkImageLogger` rely on.

:class:`HRCachedTrainDataset` also centralises the raw-HR LMDB wiring shared
verbatim by SRCNN's and SRResNet's train datasets — it lives here, not in the
torch-free :mod:`~sisr.datasets.hr_cache`, because it subclasses
:class:`torch.utils.data.Dataset`.
"""

import abc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..utils.cache import LMDBCache, LMDBCacheBuildContext
from . import derived_cache
from .hr_cache import CACHE_NAME, FORMAT_TAG, HEADER, compute_checksum, estimate_map_size


class SRDataset(torch.utils.data.Dataset, abc.ABC):
    """Abstract base for single-image super-resolution datasets.

    Subclasses call :meth:`_index_images` in ``__init__`` to populate
    :attr:`img_paths`, and implement :meth:`__len__` / :meth:`__getitem__`.
    HR is always discovered and loaded as RGB; colorspace selection happens
    downstream in :class:`~sisr.training.SRLightning` via the processor.

    Attributes:
        IMAGE_EXTENSIONS: Allowlisted lower-case file suffixes. Files with any
            other suffix (or none) are skipped by :meth:`_index_images` so a
            stray ``notes.txt`` / ``checksum.json`` never enters the manifest.
        img_dir: Directory the images were discovered in.
        img_paths: Sorted list of image file paths — the filename contract
            consumed by the datamodule and the benchmark logger.
    """

    IMAGE_EXTENSIONS: frozenset[str] = frozenset(
        {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
            ".ppm",
            ".pgm",
        }
    )

    img_dir: Path
    img_paths: list[Path]

    def _index_images(self, img_dir: str | Path) -> None:
        """Discover image files under ``img_dir`` and store the sorted manifest.

        Only files whose lower-cased suffix is in :attr:`IMAGE_EXTENSIONS`
        are kept — an allowlist rather than the old ``glob('*.*')``, which
        matched non-image files and dropped extensionless ones by accident.

        Raises:
            ValueError: If no allowlisted image files are found in ``img_dir``.
        """
        self.img_dir = Path(img_dir)
        self.img_paths = sorted(
            p
            for p in self.img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        if not self.img_paths:
            raise ValueError(f"No images found in {img_dir}")

    @staticmethod
    def _load_rgb(path: str | Path) -> np.ndarray:
        """Load an image file as an ``(H, W, 3)`` uint8 RGB array."""
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        """Converts a ``uint8`` HWC array to a ``float32`` CHW ``[0, 1]`` tensor.

        Byte-identical replacement for the removed AlbumentationsX
        ``ToFloat(255.0)`` + ``ToTensorV2()`` chain (probed against
        albumentations 2.1.0 with this project's exact call-site parameters).
        """
        return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of items the dataset serves."""

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Return the ``(lr_tensor, hr_tensor)`` pair at ``idx``.

        Paired train/validation datasets return the tuple; prediction-only
        datasets (no ground-truth HR to pair with) return a lone LR tensor —
        see :class:`~sisr.datasets.predict.PredictDataset`.
        """


class HRCachedTrainDataset(SRDataset):
    """Shared base for train datasets backed by the raw-HR LMDB cache.

    Owns the file-manifest checksum, per-image size probe, and
    :class:`~sisr.utils.cache.LMDBCache` construction/build that SRCNN's and
    SRResNet's train datasets need identically — so the two cannot drift on
    it (e.g. two different build progress-bar labels for what is, by design,
    one shared cache). A subclass supplies only its indexing scheme (grid vs
    random crop, via :attr:`_img_sizes`).

    It owns **two** caches. The raw-HR one is shared verbatim across
    architectures and scales. The derived-plane one
    (:mod:`~sisr.datasets.derived_cache`) is per ``(kind, scale)`` and holds the
    whole image already degraded, so a crop taken from it matches what the
    reference pipelines produce — degrading the crop instead resolves the
    bicubic kernel's edge taps against the crop border. Read them with
    :meth:`_read_hr` and :meth:`_read_derived`.

    Args:
        img_dir: Directory of HR images.
        scale: Upscaling factor the derived plane is built for.
        derived_kind: Which plane to derive — see
            :data:`~sisr.datasets.derived_cache.KINDS` — or ``None`` to build
            no derived cache at all. ``None`` is not a convenience: a plane
            that pushes the working set past RAM makes shuffled training
            I/O-bound, which is measurably far worse than the per-item work
            it removes. See :mod:`sisr.datasets.srcnn`.
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
        scale: int,
        derived_kind: str | None,
        use_tqdm: bool = False,
        cache_dir: str | Path | None = None,
        build_num_workers: int | None = None,
    ):
        super().__init__()

        self._index_images(img_dir)
        self.build_num_workers = build_num_workers
        self._img_sizes = self._collect_sizes()
        self._derived_kind = derived_kind

        cache_dir = Path(cache_dir) if cache_dir else self.img_dir / ".lmdb_cache"
        self._cache = LMDBCache(
            cache_dir=cache_dir,
            name=CACHE_NAME,
            checksum=self._compute_checksum(),
            length=len(self.img_paths),
            map_size=estimate_map_size(self._img_sizes),
            metadata={"format": FORMAT_TAG},
            build_fn=self._build,
            use_tqdm=use_tqdm,
        )
        # A sibling database in the same directory, not a second value in the one
        # above: that cache's checksum deliberately ignores every derivation
        # parameter, so a scale-dependent plane stored there would never be
        # invalidated by a scale change.
        self._derived = (
            None
            if derived_kind is None
            else LMDBCache(
                cache_dir=cache_dir,
                name=derived_cache.cache_name(derived_kind, scale),
                checksum=derived_cache.compute_checksum(self.img_paths, derived_kind, scale),
                length=len(self.img_paths),
                map_size=derived_cache.estimate_map_size(self._img_sizes, derived_kind, scale),
                metadata={"format": derived_cache.cache_name(derived_kind, scale)},
                build_fn=lambda ctx: ctx.parallel_build(
                    items=self.img_paths,
                    process_fn=derived_cache.process_derived_image,
                    process_args=[(i, derived_kind, scale) for i in range(len(self.img_paths))],
                    num_workers=self.build_num_workers,
                    desc=f"Building {derived_kind} cache (x{scale})",
                ),
                use_tqdm=use_tqdm,
            )
        )

    def _compute_checksum(self) -> str:
        """Computes a SHA-256 checksum over the file manifest only.

        None of a subclass's crop/grid/scale parameters enter the hash — the
        cache stores whole raw images, unaffected by anything derived from
        them at read time. Delegates to
        :func:`~sisr.datasets.hr_cache.compute_checksum`.

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

    def _parallel_build_hr(
        self,
        ctx: LMDBCacheBuildContext,
        process_fn: Callable[[Path, int], list[tuple[str, bytes]]],
    ) -> None:
        """Runs the shared HR-decode build over :attr:`img_paths`.

        *process_fn* is passed in rather than imported here so each
        architecture module keeps its own patchable module-level
        ``process_hr_image`` reference for its tests.
        """
        ctx.parallel_build(
            items=self.img_paths,
            process_fn=process_fn,
            process_args=[(i,) for i in range(len(self.img_paths))],
            num_workers=self.build_num_workers,
            desc="Building HR cache",
        )

    @abc.abstractmethod
    def _build(self, ctx: LMDBCacheBuildContext) -> None:
        """Populates the LMDB cache; subclasses delegate to :meth:`_parallel_build_hr`."""

    @contextmanager
    def _read_hr(self, img_idx: int) -> Iterator[np.ndarray]:
        """Yields the cached HR image at ``img_idx`` as an ``(H, W, 3)`` uint8 view.

        A zero-copy :meth:`~sisr.utils.cache.LMDBCache.get_buffer` view, valid only
        inside this ``with`` block — slice and ``.copy()`` out whatever is
        needed before it exits. ``(h, w)`` come from this image's own header,
        never a fixed constant, so non-square and varied-size images are safe.

        Raises:
            KeyError: If the backing LMDB entry is missing (a corrupt or
                incomplete cache).
        """
        key = f"hr_{img_idx:08d}"
        with self._cache.get_buffer(key) as buf:
            if buf is None:
                raise KeyError(key)
            h, w = HEADER.unpack_from(buf, 0)
            yield np.frombuffer(buf, dtype=np.uint8, offset=HEADER.size).reshape(h, w, 3)

    @contextmanager
    def _read_derived(self, img_idx: int) -> Iterator[np.ndarray]:
        """Yields the cached derived plane at ``img_idx`` as an ``(H, W, 3)`` uint8 view.

        Same zero-copy contract as :meth:`_read_hr` — the view is valid only
        inside the ``with`` block. The plane was derived from the **whole**
        image, so a slice of it is what the reference pipelines produce and a
        degraded slice of :meth:`_read_hr` is not.

        Raises:
            KeyError: If the backing LMDB entry is missing (a corrupt or
                incomplete cache).
        """
        if self._derived is None:
            raise RuntimeError(
                f"{type(self).__name__} was built with derived_kind=None, so there is "
                f"no derived plane to read. Either pass a kind or derive at read time."
            )
        key = f"{self._derived_kind}_{img_idx:08d}"
        with self._derived.get_buffer(key) as buf:
            if buf is None:
                raise KeyError(key)
            h, w = derived_cache.HEADER.unpack_from(buf, 0)
            yield np.frombuffer(buf, dtype=np.uint8, offset=derived_cache.HEADER.size).reshape(
                h, w, 3
            )
