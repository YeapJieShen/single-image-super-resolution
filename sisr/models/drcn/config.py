"""DRCN-paper training and evaluation defaults.

Reference: Deeply-Recursive Convolutional Network for Image Super-Resolution
(https://arxiv.org/abs/1511.04491).
"""

from dataclasses import dataclass, field
from typing import Literal

from sisr.models.base import SRModel
from sisr.processors.base import SRProcessor
from sisr.training.config import SREvalConfig, SRTrainingConfig


@dataclass
class DRCNTrainingConfig(SRTrainingConfig):
    """DRCN-paper training defaults.

    Y channel (pair with ``YChannelProcessor``). ``init_strategy='paper'``
    applies :meth:`DRCN.reset_parameters`: He init plus an identity-initialised
    recursive conv.

    The paper trains with ``L = alpha * l_recursive + (1 - alpha) * l_final +
    beta * ||theta||^2`` (``beta = 1e-4``, SGD momentum 0.9, lr 0.01), and stabilises
    training with gradient clipping. It does **not** state how ``alpha`` evolves
    during training. **Assumption, recorded:** the template ramps ``alpha`` linearly
    from 1 to 0 over the run (two ``LossWeightScheduler`` callbacks, one per term).
    ``beta`` is the optimizer's ``weight_decay``; the clip threshold is the
    trainer's ``gradient_clip_val`` with ``scale_clip_with_lr``.

    ``scale`` is left at the inherited ``None``: the model is resolution-preserving
    and the paper reports x2/x3/x4.
    """

    init_strategy: Literal["default", "paper"] = "paper"

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with DRCN's ``in_out_channels``/processor correlation.

        Args:
            model: The constructed :class:`~sisr.models.drcn.DRCN`.
            processor: The processor paired with it.

        Raises:
            ValueError: If ``model``'s ``in_out_channels`` doesn't match
                ``processor.model_channels``.
        """
        in_out_channels = model.hparams["in_out_channels"]
        if in_out_channels != processor.model_channels:
            raise ValueError(
                f"DRCN in_out_channels={in_out_channels} does not match "
                f"{type(processor).__name__}.model_channels={processor.model_channels}; "
                f"pick a processor whose model_channels matches (YChannelProcessor for 1, "
                f"RGBProcessor or YCbCrProcessor for 3)."
            )
        super().validate_against(model, processor)


@dataclass
class DRCNEvalConfig(SREvalConfig):
    """DRCN-paper eval defaults.

    Args:
        crop_border: ``None`` -- derive the field convention's ``scale`` pixels
            from the model. Resolved by ``SRLightning``.
        psnr_channels: ``['Y']`` -- the paper reports PSNR on luma.
    """

    crop_border: int | None = None
    psnr_channels: list[str] = field(default_factory=lambda: ["Y"])
