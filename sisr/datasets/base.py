"""Shared base for SR datasets — enforces the ``.img_paths`` contract.

All four architecture datasets (SRCNN / SRResNet train + validation) subclass
:class:`SRDataset`. It centralises the file discovery (extension-allowlisted
glob + empty-directory guard), the RGB image load, the ``uint8`` HWC →
``float32`` CHW ``[0, 1]`` tensor adapter, and declares the ``.img_paths``
filename contract that :class:`~sisr.training.SRDataModule` and
:class:`~sisr.training.callbacks.BenchmarkImageLogger` rely on.
"""

import abc
from pathlib import Path

import numpy as np
import torch
from PIL import Image


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
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(lr_tensor, hr_tensor)`` pair at ``idx``."""
