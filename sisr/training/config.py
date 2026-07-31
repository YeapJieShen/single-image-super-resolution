"""Per-architecture training and evaluation configuration dataclasses.

Split into two classes by lifecycle:

* ``SRTrainingConfig`` controls behaviour during ``cli fit`` — per-Conv2d
  learning rates, paper-init knobs, and the example input shape used to log
  the model graph to TensorBoard.
* ``SREvalConfig`` controls validation/test metric computation —
  boundary-pixel exclusion (``crop_border``) and which colorspaces PSNR is
  reported in.

Per-architecture defaults live in subclasses next to the model code (e.g.
``sisr.models.srcnn.SRCNNTrainingConfig``); a YAML user picks them via
``class_path`` on ``model.training_config`` / ``model.eval_config`` and
overrides individual fields with ``init_args``.

The colorspace the model trains in is no longer a string field here; it is
expressed by the choice of processor (see ``sisr.processors``).

``SRTrainingConfig.validate_against`` / ``SREvalConfig.psnr_keys`` are the
config-side half of the correlated-field validation seam: fields like
``num_channels`` (model) and ``class_path`` (processor) live in sibling
objects a single dataclass ``__post_init__`` cannot see across, so the
cross-object check happens once both are constructed, orchestrated by
``SRLightning.__init__``.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch

from ..models.base import SRModel
from ..processors.base import SRProcessor

# Per-colorspace channel names, in report order. Doubles as the set of
# supported ``psnr_channels`` entries (validated in `SREvalConfig.__post_init__`).
_PSNR_CHANNEL_NAMES: dict[str, tuple[str, ...]] = {
    "RGB": ("R", "G", "B"),
    "YCbCr": ("Y", "Cb", "Cr"),
}


@dataclass
class SRTrainingConfig:
    """How to train the SR model — affects optimizer setup and weight init.

    Args:
        layer_lrs: Absolute per-``Conv2d`` learning rates (one entry per
            ``Conv2d`` in the model, in module-traversal order).  When
            ``None`` (default), training uses the optimizer's base ``lr``
            uniformly across all parameters.  Only valid for architectures
            where every trainable parameter lives in a ``Conv2d`` (no
            BatchNorm / PReLU); ``SRLightning.configure_optimizers``
            raises ``ValueError`` otherwise.

        example_input_shape: Shape of a single input sample *excluding* the
            batch dimension (e.g. ``(1, 33, 33)`` for a 33×33 Y-channel patch).
            When provided, ``self.example_input_array`` is set so the
            TensorBoard logger can capture the model graph, and
            :meth:`validate_against` probes the model with it.

        init_strategy: ``'paper'`` triggers a paper-faithful weight init via
            ``SRModel.reset_parameters`` in ``SRLightning``'s constructor;
            ``'default'`` (the default) skips it and uses PyTorch's defaults.
            Subclasses pin a paper-faithful default (e.g. ``SRCNNTrainingConfig``
            uses ``'paper'``).

        init_mean: Mean of the Gaussian used by SRCNN's
            ``init_strategy='paper'``. Other paper-init implementations may
            ignore this. Defaults to ``0.0``.

        init_std: Std of the Gaussian used by SRCNN's
            ``init_strategy='paper'``. Other paper-init implementations may
            ignore this. Defaults to ``0.01``.
    """

    layer_lrs: list[float] | None = None
    example_input_shape: tuple[int, ...] | None = None
    init_strategy: Literal["default", "paper"] = "default"
    init_mean: float = 0.0
    init_std: float = 0.01

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Validate this config against the model/processor it will pair with.

        Universal, architecture-agnostic checks: when ``example_input_shape``
        is set, its channel dimension must equal ``processor.model_channels``,
        and a ``torch.no_grad()`` forward pass of the real ``model`` on a
        zero tensor of that shape must succeed. The probe exercises the
        actual ``nn.Module`` rather than a separate description of it, so it
        cannot go stale as the architecture evolves. A no-op when
        ``example_input_shape`` is unset — it is optional (TensorBoard graph
        / FLOPs reporting only).

        Subclasses (e.g. ``SRCNNTrainingConfig``) override this to add
        architecture-specific correlation checks — e.g. ``num_channels`` vs
        ``processor.model_channels`` — that exist purely to raise an
        actionable message *before* a mismatched pairing would otherwise
        surface as a raw shape-mismatch error from this probe. They call
        ``super().validate_against(model, processor)`` to keep the universal
        checks.

        Args:
            model: The constructed :class:`~sisr.models.base.SRModel` this
                config will train/evaluate.
            processor: The :class:`~sisr.processors.base.SRProcessor` paired
                with ``model`` for this run.

        Raises:
            ValueError: If ``example_input_shape`` is set and its channel
                dimension doesn't match ``processor.model_channels``.
        """
        if self.example_input_shape is None:
            return
        channels = self.example_input_shape[0]
        if channels != processor.model_channels:
            raise ValueError(
                f"training_config.example_input_shape has {channels} channel(s) "
                f"(position 0), but {type(processor).__name__}.model_channels="
                f"{processor.model_channels}. Fix example_input_shape[0] or pick "
                f"a processor whose model_channels matches the model's actual "
                f"input/output channel count."
            )
        dummy = torch.zeros(1, *self.example_input_shape)
        with torch.no_grad():
            model(dummy)


@dataclass
class SREvalConfig:
    """How to compute validation/test metrics — affects scoring only, not training.

    Args:
        crop_border: Number of border pixels to exclude on each edge before
            computing PSNR / SSIM.  Standard SR-evaluation convention is to
            crop the outer ``scale`` pixels (e.g. ``crop_border=3`` for x3).

        psnr_channels: Colorspaces in which PSNR is reported.  Supported
            values are ``'RGB'`` and ``'YCbCr'``.  Multiple entries are
            allowed (e.g. ``['RGB', 'YCbCr']`` produces both
            ``val_psnr(RGB)`` and ``val_psnr(YCbCr)``).

        separate_psnr: When ``True``, also reports PSNR for each individual
            channel within each requested colorspace (e.g. ``'RGB'`` adds
            ``val_psnr(R)`` / ``val_psnr(G)`` / ``val_psnr(B)`` alongside
            the aggregate ``val_psnr(RGB)``).
    """

    crop_border: int = 0
    psnr_channels: list[str] = field(default_factory=lambda: ["RGB"])
    separate_psnr: bool = False

    def __post_init__(self) -> None:
        """Validate ``psnr_channels`` at construction.

        Raises:
            ValueError: If any entry is not a supported colorspace
                (``'RGB'`` or ``'YCbCr'``).
        """
        valid = tuple(_PSNR_CHANNEL_NAMES)
        invalid = [c for c in self.psnr_channels if c not in valid]
        if invalid:
            raise ValueError(
                f"SREvalConfig.psnr_channels entries must be one of "
                f"{list(valid)}; got unsupported {invalid}. Fix "
                f"model.eval_config.init_args.psnr_channels in your YAML."
            )

    @property
    def psnr_keys(self) -> list[str]:
        """Ordered PSNR metric keys this config requests.

        For each colorspace in ``psnr_channels`` (in order), per-channel keys
        are emitted first when ``separate_psnr`` is ``True``, followed by the
        aggregate colorspace key itself — e.g. ``psnr_channels=['RGB']`` with
        ``separate_psnr=True`` yields ``['R', 'G', 'B', 'RGB']``.

        This is the seam consumed by ``SRLightning`` (val metric logging /
        HParams registration) and ``BenchmarkImageLogger`` (benchmark PSNR
        key selection) — the single place the key set and its order are
        derived, so the two consumers cannot disagree.

        Returns:
            Ordered list of PSNR keys, e.g. ``['RGB', 'YCbCr']`` or
            ``['R', 'G', 'B', 'RGB', 'Y', 'Cb', 'Cr', 'YCbCr']``.
        """
        keys: list[str] = []
        for cs in self.psnr_channels:
            if self.separate_psnr:
                keys.extend(_PSNR_CHANNEL_NAMES[cs])
            keys.append(cs)
        return keys
