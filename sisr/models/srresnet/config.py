"""SRResNet-paper-faithful training and evaluation defaults.

Subclassing :class:`~sisr.training.config.SRTrainingConfig` /
:class:`~sisr.training.config.SREvalConfig` lets the SRResNet recipe live
alongside the model architecture, so an SRResNet experiment YAML only has
to point at these classes via ``class_path`` to inherit defaults.

Reference: Photo-Realistic Single Image Super-Resolution Using a
Generative Adversarial Network (https://arxiv.org/pdf/1609.04802),
Section 3.2 (SRResNet baseline).
"""

from dataclasses import dataclass, field
from typing import Literal

from sisr.models.base import SRModel
from sisr.processors.base import SRProcessor
from sisr.training.config import SREvalConfig, SRTrainingConfig


@dataclass
class SRResNetTrainingConfig(SRTrainingConfig):
    """SRResNet-paper-faithful training defaults.

    The paper trains on full RGB (selected at the YAML layer by pairing
    with :class:`~sisr.processors.RGBProcessor`) using Adam (lr 1e-4) and
    no per-layer LRs.

    The paper does not specify a weight-initialization scheme.
    ``init_strategy='paper'`` is reserved for a future PR that wires up a
    project-chosen init (e.g. Kaiming-style for the PReLU activations) via
    :meth:`SRResNet.reset_parameters`. Today the field defaults to
    ``'default'`` so SRResNet ships with PyTorch's built-in init; flipping
    to ``'paper'`` is currently a no-op because
    :meth:`SRModel.reset_parameters` (the inherited base) does nothing.
    The field exists now so a future PR can add the implementation without
    a YAML schema change.
    """

    init_strategy: Literal["default", "paper"] = "default"

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with SRResNet's ``in_out_channels``/processor correlation.

        Raises a readable error before the base's forward probe would
        otherwise surface the same defect as a cryptic Conv2d shape mismatch.

        Args:
            model: The constructed :class:`~sisr.models.srresnet.SRResNet` instance.
            processor: The :class:`~sisr.processors.base.SRProcessor` paired
                with ``model``.

        Raises:
            ValueError: If ``model``'s ``in_out_channels`` doesn't match
                ``processor.model_channels``.
        """
        in_out_channels = model.hparams["in_out_channels"]
        if in_out_channels != processor.model_channels:
            raise ValueError(
                f"SRResNet in_out_channels={in_out_channels} does not match "
                f"{type(processor).__name__}.model_channels={processor.model_channels}. "
                f"in_out_channels sets both the head Conv2d's input and the tail "
                f"Conv2d's output (after the scale={model.hparams['scale']}x "
                f"PixelShuffle upsampling); pick a processor whose model_channels "
                f"matches (e.g. RGBProcessor or YCbCrProcessor for in_out_channels=3)."
            )
        super().validate_against(model, processor)


@dataclass
class SRResNetEvalConfig(SREvalConfig):
    """SRResNet-paper-faithful eval defaults.

    Reports PSNR on RGB and Y — Ledig et al. quote Y-channel only (the full
    YCbCr aggregate reads optimistically high since chroma planes are far
    smoother than luma); excludes the outer ``scale=4`` pixels before
    computing PSNR / SSIM, matching the standard SR-evaluation convention.

    Args:
        crop_border: Overrides the base default to ``4`` (outer pixels
            excluded before PSNR / SSIM at the standard ``x4`` scale).
        psnr_channels: Overrides the base default to ``['RGB', 'Y']`` —
            ``'Y'`` is the paper's own metric; ``'RGB'`` is a supplementary
            aggregate.
    """

    crop_border: int = 4
    psnr_channels: list[str] = field(default_factory=lambda: ["RGB", "Y"])
