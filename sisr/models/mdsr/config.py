"""MDSR training and evaluation defaults.

Reference: Enhanced Deep Residual Networks for Single Image Super-Resolution
(https://arxiv.org/abs/1707.02921), Section 4.
"""

from dataclasses import dataclass, field
from typing import Literal

from sisr.models.base import SRModel
from sisr.processors.base import SRProcessor
from sisr.training.config import SREvalConfig, SRTrainingConfig


@dataclass
class MDSRTrainingConfig(SRTrainingConfig):
    """MDSR training defaults: one model over ``scale=[2, 3, 4]``, L1 loss, Adam 1e-4.

    Each batch draws one scale (``scale`` is a list, so the data pipeline samples per batch);
    the optimizer, schedule and batch size live in the YAML. ``init_strategy`` is ``'default'``
    because the paper states no initialisation.
    """

    scale: int | list[int] | None = field(default_factory=lambda: [2, 3, 4])
    init_strategy: Literal["default", "paper"] = "default"

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with the model/processor channel correlation."""
        in_out_channels = model.hparams["in_out_channels"]
        if in_out_channels != processor.model_channels:
            raise ValueError(
                f"MDSR in_out_channels={in_out_channels} does not match "
                f"{type(processor).__name__}.model_channels={processor.model_channels}; pick a "
                f"processor whose model_channels matches (RGBProcessor for 3)."
            )
        super().validate_against(model, processor)


@dataclass
class MDSREvalConfig(SREvalConfig):
    """MDSR's reporting: PSNR/SSIM on Y, each sample shaved by its own scale, Wang SSIM.

    ``crop_border=None`` stays ``None`` for several scales: the border is each sample's scale.
    As for EDSR, the paper's table used MATLAB's 8-bit luma, which this pipeline's float Y reads
    0.01-0.10 dB above (see :class:`~sisr.models.edsr.EDSREvalConfig`).
    """

    crop_border: int | None = None
    psnr_channels: list[str] = field(default_factory=lambda: ["Y"])
