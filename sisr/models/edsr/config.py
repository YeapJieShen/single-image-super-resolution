"""EDSR training and evaluation defaults.

Reference: Enhanced Deep Residual Networks for Single Image Super-Resolution
(https://arxiv.org/abs/1707.02921), Sections 3-4.
"""

from dataclasses import dataclass, field
from typing import Literal

from sisr.models.base import SRModel
from sisr.processors.base import SRProcessor
from sisr.training.config import SREvalConfig, SRTrainingConfig


@dataclass
class EDSRTrainingConfig(SRTrainingConfig):
    """EDSR training defaults: L1 loss, Adam 1e-4 halved every 2e5 updates, batch 16, 48x48 LR.

    The optimizer, schedule and batch size live in the YAML. ``init_strategy`` is ``'default'``
    because the paper states no initialisation; ``scale`` defaults to 2 (the paper trains x3 and
    x4 from the x2 model, a trick this repo mentions and does not implement).
    """

    scale: int | list[int] | None = 2
    init_strategy: Literal["default", "paper"] = "default"

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with the model/processor channel correlation."""
        in_out_channels = model.hparams["in_out_channels"]
        if in_out_channels != processor.model_channels:
            raise ValueError(
                f"EDSR in_out_channels={in_out_channels} does not match "
                f"{type(processor).__name__}.model_channels={processor.model_channels}; pick a "
                f"processor whose model_channels matches (RGBProcessor for 3)."
            )
        super().validate_against(model, processor)


@dataclass
class EDSREvalConfig(SREvalConfig):
    """EDSR's reporting: PSNR/SSIM on Y, a border of ``scale``, Wang SSIM.

    The authors' own evaluation shaves ``scale`` pixels on benchmark sets and takes Y from the
    studio-range BT.601 weights, which is what ``psnr/val/Y`` does. **The paper's table was
    produced with MATLAB** (sec 4.5), whose ``rgb2ycbcr`` on 8-bit images returns 8-bit luma,
    so both planes are rounded; this pipeline scores the float output's float Y, which reads
    0.01-0.10 dB higher on the released checkpoints. Under the MATLAB protocol they match the
    paper's EDSR and EDSR+ rows to within 0.006 dB. A single-channel image is replicated to
    three channels before the transform, as the authors' loader does, so (unlike the MATLAB
    releases of VDSR and DRCN) there is no separate grey-image rule.
    """

    crop_border: int | None = None
    psnr_channels: list[str] = field(default_factory=lambda: ["Y"])
