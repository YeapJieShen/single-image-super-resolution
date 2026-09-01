"""SRResNet-paper-faithful training and evaluation defaults.

Subclassing the base configs keeps the recipe beside the architecture, so an
experiment YAML inherits it by naming these classes in ``class_path``.

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

    Full RGB, Adam at lr 1e-4, no per-layer LRs. Colorspace *and* intensity
    range are chosen by the processor in YAML:
    :class:`~sisr.processors.RGBSignedOutputProcessor` for the paper's LR
    ``[0, 1]`` / HR ``[-1, 1]`` asymmetry, :class:`~sisr.processors.RGBProcessor`
    for plain ``[0, 1]``.

    **The paper specifies no weight init**, so ``init_strategy`` defaults to
    ``'default'``. Setting ``'paper'`` is a no-op today: the inherited
    :meth:`SRModel.reset_parameters` does nothing. The field exists so an
    implementation can land without a YAML schema change.

    ``scale`` defaults to ``4`` -- the baseline reproduced here is fixed at one
    factor, unlike SRCNN, whose model carries no ``scale`` hparam at all. It is
    validated against the model, not merely recorded.
    """

    init_strategy: Literal["default", "paper"] = "default"
    scale: int = 4

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with SRResNet's ``in_out_channels``/processor correlation.

        Fires before the base's forward probe, which would surface the same
        defect as a cryptic Conv2d shape mismatch.

        Args:
            model: The constructed ``SRResNet``.
            processor: The processor paired with it.

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

    PSNR on RGB and Y (Ledig et al. quote Y only; the YCbCr aggregate reads
    optimistically high, chroma being far smoother than luma), the field's
    border convention, and the daala SSIM the paper itself used.

    Args:
        crop_border: ``None`` -- derive the field convention's ``scale`` pixels
            from the model rather than pin a constant. Resolved by
            ``SRLightning``; for this config that is 4.
        psnr_channels: ``['RGB', 'Y']`` -- ``'Y'`` is the paper's metric,
            ``'RGB'`` a supplementary aggregate.
        ssim_impl: ``'daala'``, whose gaussian sigma scales with image height
            rather than Wang's fixed 11x11. **SSIM under this setting is
            comparable to the paper and not to the wider SR literature**, which
            reports Wang. PSNR is unaffected, being implementation-invariant.
            See :mod:`sisr.metrics.ssim`.
    """

    crop_border: int | None = None
    psnr_channels: list[str] = field(default_factory=lambda: ["RGB", "Y"])
    ssim_impl: Literal["wang", "daala"] = "daala"
