"""ESRGAN training and evaluation defaults.

Reference: ESRGAN (https://arxiv.org/abs/1809.00219), Section 4.1.
"""

from dataclasses import dataclass, field
from typing import Literal

from sisr.models.base import SRModel
from sisr.processors.base import SRProcessor
from sisr.training.config import AdversarialTrainingConfig, SREvalConfig, SRTrainingConfig


def _check_channels(model: SRModel, processor: SRProcessor) -> None:
    in_out_channels = model.hparams["in_out_channels"]
    if in_out_channels != processor.model_channels:
        raise ValueError(
            f"RRDBNet in_out_channels={in_out_channels} does not match "
            f"{type(processor).__name__}.model_channels={processor.model_channels}; pick a "
            f"processor whose model_channels matches (RGBProcessor for 3)."
        )


@dataclass
class RRDBNetTrainingConfig(SRTrainingConfig):
    """PSNR-stage defaults: L1 loss, Adam 2e-4 halved every 200k updates, batch 16, 128 crops.

    ``init_strategy='paper'`` applies :meth:`RRDBNet.reset_parameters` (the smaller init).
    The learning-rate schedule, optimizer and batch size live in the YAML.
    """

    scale: int | list[int] | None = 4
    init_strategy: Literal["default", "paper"] = "paper"

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with the model/processor channel correlation."""
        _check_channels(model, processor)
        super().validate_against(model, processor)


@dataclass
class ESRGANTrainingConfig(AdversarialTrainingConfig):
    """GAN-stage defaults: ``adversarial_weight`` is the paper's lambda, 5e-3.

    The generator starts from the PSNR-stage artifact via ``init_from``. The rest of the
    paper's loss (VGG54 before activation weight 1, L1 weight eta = 1e-2) is the criterion.
    """

    scale: int | list[int] | None = 4
    init_strategy: Literal["default", "paper"] = "paper"
    adversarial_weight: float = 5e-3

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with the model/processor channel correlation."""
        _check_channels(model, processor)
        super().validate_against(model, processor)


@dataclass
class RRDBNetEvalConfig(SREvalConfig):
    """ESRGAN's reporting: PSNR/SSIM on Y, border = scale, Wang SSIM.

    The paper reports "PSNR/SSIM on Y channel"; a border of ``scale`` pixels (the field
    convention) is an assumption, as is ``ssim_impl='wang'``.
    """

    crop_border: int | None = None
    psnr_channels: list[str] = field(default_factory=lambda: ["Y"])


@dataclass
class ESRGANEvalConfig(RRDBNetEvalConfig):
    """The GAN stage also reports ``lpips`` and ``dists``: PSNR/SSIM fall under an adversarial loss.

    ``lpips`` needs the ``[perceptual]`` extra.
    """

    perceptual_metrics: list[str] = field(default_factory=lambda: ["lpips", "dists"])
