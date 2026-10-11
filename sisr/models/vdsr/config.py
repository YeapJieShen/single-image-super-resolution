"""VDSR-paper training and evaluation defaults.

Reference: Accurate Image Super-Resolution Using Very Deep Convolutional Networks
(https://arxiv.org/abs/1511.04587).
"""

from dataclasses import dataclass, field
from typing import Literal

from sisr.models.base import SRModel
from sisr.processors.base import SRProcessor
from sisr.training.config import SREvalConfig, SRTrainingConfig


@dataclass
class VDSRTrainingConfig(SRTrainingConfig):
    """VDSR-paper training defaults.

    Y channel (pair with ``YChannelProcessor``). ``init_strategy='paper'`` applies
    :meth:`VDSR.reset_parameters` (He init).

    The paper trains one multi-scale model on 41x41 sub-images from x2/x3/x4 together
    (a batch mixes scales): batch 64, SGD momentum 0.9, weight decay 1e-4, learning
    rate 0.1 divided by 10 every 20 epochs for 80 epochs, and *adjustable gradient
    clipping* to ``[-theta/lr, theta/lr]`` (``scale_clip_with_lr``). It states no value
    for ``theta``, so the template's is this project's assumption. Set ``scale`` to the
    list ``[2, 3, 4]`` for the multi-scale model.
    """

    init_strategy: Literal["default", "paper"] = "paper"

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with VDSR's ``in_out_channels``/processor correlation.

        Args:
            model: The constructed :class:`~sisr.models.vdsr.VDSR`.
            processor: The processor paired with it.

        Raises:
            ValueError: If ``model``'s ``in_out_channels`` doesn't match
                ``processor.model_channels``.
        """
        in_out_channels = model.hparams["in_out_channels"]
        if in_out_channels != processor.model_channels:
            raise ValueError(
                f"VDSR in_out_channels={in_out_channels} does not match "
                f"{type(processor).__name__}.model_channels={processor.model_channels}; "
                f"pick a processor whose model_channels matches (YChannelProcessor for 1, "
                f"RGBProcessor or YCbCrProcessor for 3)."
            )
        super().validate_against(model, processor)


@dataclass
class VDSREvalConfig(SREvalConfig):
    """VDSR-paper eval defaults.

    Args:
        crop_border: ``None`` -- the field convention's ``scale`` pixels, from each
            sample's own scale for a multi-scale run. Resolved by ``SRLightning``.
        psnr_channels: ``['Y']`` -- the paper reports PSNR on luma.
    """

    crop_border: int | None = None
    psnr_channels: list[str] = field(default_factory=lambda: ["Y"])
