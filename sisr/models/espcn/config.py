"""ESPCN-paper training and evaluation defaults.

Reference: Real-Time Single Image and Video Super-Resolution Using an Efficient
Sub-Pixel Convolutional Neural Network (https://arxiv.org/pdf/1609.05158),
Section 3.2.
"""

from dataclasses import dataclass, field
from typing import Literal

from sisr.models.base import SRModel
from sisr.processors.base import SRProcessor
from sisr.training.config import SREvalConfig, SRTrainingConfig


def _paper_layer_lrs() -> list[float | list[float]]:
    """Constant per-conv LRs: 1e-2 for the first two, a tenth of that for the last.

    A function rather than a lambda so the element type is ``float | list[float]``
    (``list`` is invariant, so an inferred ``list[float]`` will not satisfy the field).
    """
    return [1.0e-2, 1.0e-2, 1.0e-3]


@dataclass
class ESPCNTrainingConfig(SRTrainingConfig):
    """ESPCN-paper training defaults.

    Y channel (pair with ``YChannelProcessor``), MSE. The paper starts at lr 0.01
    and decays it to 0.0001 when the cost stops improving, and "the final layer
    learns 10 times slower". **Only the second is reproduced:** ``layer_lrs`` is
    constant per layer, ``[1e-2, 1e-2, 1e-3]``, with no decay. Deviation recorded.

    **The paper specifies no weight init**, so ``init_strategy`` defaults to
    ``'default'``; ``'paper'`` is a no-op today because the inherited
    :meth:`SRModel.reset_parameters` does nothing.

    ``scale`` defaults to ``3``, the factor the paper leads with (it also reports
    x4); templates set it explicitly. It is validated against the model.

    **Not yet reproduced:** the paper's LR images come from a Gaussian blur of
    unstated sigma; this project's bicubic degradation is used instead.
    """

    layer_lrs: list[float | list[float]] | None = field(default_factory=_paper_layer_lrs)
    init_strategy: Literal["default", "paper"] = "default"
    scale: int = 3

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with ESPCN's ``in_out_channels``/processor correlation.

        Args:
            model: The constructed :class:`~sisr.models.espcn.ESPCN`.
            processor: The processor paired with it.

        Raises:
            ValueError: If ``model``'s ``in_out_channels`` doesn't match
                ``processor.model_channels``.
        """
        in_out_channels = model.hparams["in_out_channels"]
        if in_out_channels != processor.model_channels:
            raise ValueError(
                f"ESPCN in_out_channels={in_out_channels} does not match "
                f"{type(processor).__name__}.model_channels={processor.model_channels}. "
                f"in_out_channels sets both the first Conv2d's input and the last "
                f"Conv2d's output (before the PixelShuffle); pick a processor whose "
                f"model_channels matches (YChannelProcessor for in_out_channels=1, "
                f"RGBProcessor or YCbCrProcessor for 3)."
            )
        super().validate_against(model, processor)


@dataclass
class ESPCNEvalConfig(SREvalConfig):
    """ESPCN-paper eval defaults.

    Args:
        crop_border: ``None`` -- derive the field convention's ``scale`` pixels
            from the model. Resolved by ``SRLightning``.
        psnr_channels: ``['Y']`` -- the paper reports PSNR on luma.
    """

    crop_border: int | None = None
    psnr_channels: list[str] = field(default_factory=lambda: ["Y"])
