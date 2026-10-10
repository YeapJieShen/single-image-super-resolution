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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..utils.cache import LMDBCache, LMDBCacheBuildContext
from . import derived_cache
from .hr_cache import CACHE_NAME, FORMAT_TAG, HEADER, compute_checksum, estimate_map_size


def as_scales(scale: int | Sequence[int]) -> tuple[int, ...]:
    """Normalise a dataset's ``scale`` argument to a non-empty tuple of distinct positive ints.

    The one owner of what a train dataset's ``scale`` may be: a single factor, or
    several for multi-scale training.

    Raises:
        ValueError: If it is empty, holds a non-positive or non-int entry, or repeats one.
    """
    scales = (scale,) if isinstance(scale, int) else tuple(scale)
    if (
        not scales
        or any(not isinstance(s, int) or isinstance(s, bool) or s < 1 for s in scales)
        or len(set(scales)) != len(scales)
    ):
        raise ValueError(f"scale must be a positive int or a list of distinct ones; got {scale!r}.")
    return scales


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
    def _is_grey_file(path: str | Path) -> torch.Tensor:
        """Whether ``path`` is a single-channel image on disk (PIL mode ``L``), as a 0-dim bool.

        The file mode, not the pixel values: an RGB file whose channels happen to be equal
        (Set14's ``man``) is a colour image to MATLAB, which reads it with three channels.
        """
        with Image.open(path) as img:
            return torch.tensor(img.mode == "L")

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
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Return the ``(lr_tensor, hr_tensor)`` pair at ``idx``.

        Paired train/validation datasets return the tuple (an eval dataset may append an
        ``is_grey`` flag); prediction-only
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
        scale: Upscaling factor the derived plane is built for, or several: one
            derived cache is built per scale, and :meth:`_read_derived` takes the
            scale to read. The first one is the default.
        derived_kind: Which plane to derive — see
            :data:`~sisr.datasets.derived_cache.KINDS`.
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
        scale: int | Sequence[int],
        derived_kind: str,
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
        # invalidated by a scale change. One per scale, each keyed by its own.
        scales = [scale] if isinstance(scale, int) else list(scale)
        self._derived_caches: dict[int, LMDBCache] = {
            s: self._make_derived_cache(cache_dir, derived_kind, s, use_tqdm) for s in scales
        }
        self._derived_default_scale = scales[0]

    @property
    def _derived(self) -> LMDBCache:
        """The default (first) scale's derived cache."""
        return self._derived_caches[self._derived_default_scale]

    def _make_derived_cache(
        self, cache_dir: Path, kind: str, scale: int, use_tqdm: bool
    ) -> LMDBCache:
        """Builds the sibling LMDB cache holding the *kind* plane derived at *scale*."""
        return LMDBCache(
            cache_dir=cache_dir,
            name=derived_cache.cache_name(kind, scale),
            checksum=derived_cache.compute_checksum(self.img_paths, kind, scale),
            length=len(self.img_paths),
            map_size=derived_cache.estimate_map_size(self._img_sizes, kind, scale),
            metadata={"format": derived_cache.cache_name(kind, scale)},
            build_fn=lambda ctx: ctx.parallel_build(
                items=self.img_paths,
                process_fn=derived_cache.process_derived_image,
                process_args=[(i, kind, scale) for i in range(len(self.img_paths))],
                num_workers=self.build_num_workers,
                desc=f"Building {kind} cache (x{scale})",
            ),
            use_tqdm=use_tqdm,
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
    def _read_derived(self, img_idx: int, scale: int | None = None) -> Iterator[np.ndarray]:
        """Yields the cached derived plane at ``img_idx`` as an ``(H, W, 3)`` uint8 view.

        Same zero-copy contract as :meth:`_read_hr` — the view is valid only
        inside the ``with`` block. The plane was derived from the **whole**
        image, so a slice of it is what the reference pipelines produce and a
        degraded slice of :meth:`_read_hr` is not.

        Args:
            img_idx: Index into :attr:`img_paths`.
            scale: Which scale's plane; ``None`` is the first scale given.

        Raises:
            KeyError: If the backing LMDB entry is missing (a corrupt or
                incomplete cache).
        """
        key = f"{self._derived_kind}_{img_idx:08d}"
        cache = self._derived_caches[self._derived_default_scale if scale is None else scale]
        with cache.get_buffer(key) as buf:
            if buf is None:
                raise KeyError(key)
            h, w = derived_cache.HEADER.unpack_from(buf, 0)
            yield np.frombuffer(buf, dtype=np.uint8, offset=derived_cache.HEADER.size).reshape(
                h, w, 3
            )
