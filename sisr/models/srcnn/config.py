"""SRCNN-paper-faithful training and eval defaults.

Subclassing ``SRTrainingConfig`` / ``SREvalConfig`` from
``sisr.training.config`` lets the SRCNN-paper recipe live alongside the
model architecture, so an SRCNN experiment YAML only has to point at these
classes via ``class_path`` to inherit defaults.

Reference: Image Super-Resolution Using Deep Convolutional Networks
(https://arxiv.org/pdf/1501.00092).
"""

from dataclasses import dataclass, field
from typing import Literal

from sisr.models.base import SRModel
from sisr.processors.base import SRProcessor
from sisr.training.config import SREvalConfig, SRTrainingConfig


@dataclass
class SRCNNTrainingConfig(SRTrainingConfig):
    """SRCNN-paper-faithful training defaults.

    The paper trains on the Y channel of YCbCr only (selected at the YAML
    layer by pairing with ``YChannelProcessor``) and
    uses a per-layer learning rate of ``1e-4`` for the feature-extraction
    and non-linear-mapping layers and ``1e-5`` for the reconstruction layer
    — i.e. the last layer learns 10× slower. Weight initialization follows
    the paper's Gaussian schedule (``N(0, 0.001)`` with zero biases); set
    ``init_strategy='default'`` to fall back to PyTorch's built-in init.
    Override any field in YAML to deviate.

    ``scale`` is left at the inherited ``None`` default (not overridden
    here): unlike SRResNet, ``SRCNN`` carries no ``scale`` hparam — it is
    resolution-preserving and trains on whatever scale the datamodule
    supplies — and the paper itself reports x2/x3/x4 results rather than a
    single fixed factor, so there is no one paper-correct value to pin.

    Args:
        layer_lrs: Per-``Conv2d`` LRs ``[1e-4, 1e-4, 1e-5]`` matching the
            paper's recipe. Set to ``None`` to disable per-layer LRs.
        init_strategy: ``'paper'`` (default) triggers the Gaussian
            init via ``SRCNN.reset_parameters`` in
            ``SRLightning``'s constructor;
            ``'default'`` skips it and uses PyTorch's defaults.
        init_mean: Gaussian mean for ``init_strategy='paper'``; inherited
            from ``SRTrainingConfig`` (``0.0``, matching the paper).
        init_std: Gaussian std for ``init_strategy='paper'``; overrides
            the shared base's ``0.01`` to ``0.001``, the value the SRCNN
            paper actually specifies. Override in YAML to deviate.
    """

    layer_lrs: list[float] | None = field(default_factory=lambda: [1.0e-4, 1.0e-4, 1.0e-5])
    init_strategy: Literal["default", "paper"] = "paper"
    init_std: float = 0.001

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Extend the base checks with SRCNN's ``num_channels``/processor correlation.

        Raises a readable error before the base's forward probe would
        otherwise surface the same defect as a cryptic Conv2d shape mismatch.

        Args:
            model: The constructed :class:`~sisr.models.srcnn.SRCNN` instance.
            processor: The :class:`~sisr.processors.base.SRProcessor` paired
                with ``model``.

        Raises:
            ValueError: If ``model``'s ``num_channels`` doesn't match
                ``processor.model_channels``.
        """
        num_channels = model.hparams["num_channels"]
        if num_channels != processor.model_channels:
            raise ValueError(
                f"SRCNN num_channels={num_channels} does not match "
                f"{type(processor).__name__}.model_channels={processor.model_channels}. "
                f"num_channels sets both the feature-extraction input and the "
                f"reconstruction output channel count; pick a processor whose "
                f"model_channels matches (e.g. YChannelProcessor for num_channels=1, "
                f"RGBProcessor or YCbCrProcessor for num_channels=3)."
            )
        super().validate_against(model, processor)


@dataclass
class SRCNNEvalConfig(SREvalConfig):
    """SRCNN-paper-faithful eval defaults.

    Reports PSNR on RGB and Y — the paper's own metric (Dong et al. quote
    Y-channel only; the full YCbCr aggregate reads optimistically high
    since chroma planes are far smoother than luma) — and excludes the
    outer ``scale=3`` pixels per the standard SR-evaluation convention.

    Args:
        crop_border: Overrides the base default to ``3`` (outer pixels
            excluded before PSNR / SSIM at the standard ``x3`` scale).
        psnr_channels: Overrides the base default to ``['RGB', 'Y']`` —
            ``'Y'`` is the paper's own metric; ``'RGB'`` is a supplementary
            aggregate.
    """

    crop_border: int = 3
    psnr_channels: list[str] = field(default_factory=lambda: ["RGB", "Y"])
