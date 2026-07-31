"""LR-only prediction dataset — real inference images, no HR pair.

Unlike the architecture datasets (which synthesize LR from HR for training and
evaluation), :class:`PredictDataset` loads images that already *are* LR — the
only genuine no-ground-truth input the framework serves. Reuses
:class:`~sisr.datasets.base.SRDataset` for file discovery and the uint8-HWC ->
float32-CHW tensor adapter, since both are architecture/colorspace agnostic.
"""

from pathlib import Path

import torch

from .base import SRDataset


class PredictDataset(SRDataset):
    """Loads real LR images for inference — no synthetic downsampling, no HR pair.

    Serves the raw image at each path as RGB ``float32`` in ``[0, 1]``,
    unmodified. Colorspace extraction (RGB vs. Y vs. YCbCr) happens
    downstream in :class:`~sisr.training.SRLightning` via the processor,
    exactly as it does for the paired training/validation datasets.

    Args:
        img_dir (str | Path): Directory containing the LR images to run
            inference on.

    Raises:
        ValueError: If no image files are found in ``img_dir``, or if two
            images share a filename stem.
    """

    def __init__(self, img_dir: str | Path):
        super().__init__()
        self._index_images(img_dir)
        # SRPredictionWriter names outputs by stem, so `cat.png` and `cat.jpg`
        # would silently overwrite each other. Fail at construction instead.
        stems: dict[str, Path] = {}
        for path in self.img_paths:
            if path.stem in stems:
                raise ValueError(
                    f"Duplicate filename stem {path.stem!r} in {img_dir}: "
                    f"{stems[path.stem].name} and {path.name}. Predictions are "
                    f"named by stem, so these would overwrite each other."
                )
            stems[path.stem] = path
        self._to_tensor = self._to_tensor_transform()

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns the LR RGB tensor at ``idx``.

        Args:
            idx (int): Zero-based image index.

        Returns:
            ``float32`` tensor of shape ``(3, H, W)`` in ``[0, 1]``.
        """
        arr = self._load_rgb(self.img_paths[idx])
        return self._to_tensor(image=arr)["image"]
