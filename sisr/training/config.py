"""Per-architecture training and evaluation configuration dataclasses.

Split into two classes by lifecycle:

* :class:`SRTrainingConfig` controls behaviour during ``cli fit`` — per-Conv2d
  learning rates, paper-init knobs, and the example input shape used to log
  the model graph to TensorBoard.
* :class:`SREvalConfig` controls validation/test metric computation —
  boundary-pixel exclusion (``crop_border``) and which colorspaces PSNR is
  reported in.

Per-architecture defaults live in subclasses next to the model code (e.g.
``sisr.models.srcnn.SRCNNTrainingConfig``); a YAML user picks them via
``class_path`` on ``model.training_config`` / ``model.eval_config`` and
overrides individual fields with ``init_args``.

The colorspace the model trains in is no longer a string field here; it is
expressed by the choice of processor (see :mod:`sisr.processors`).
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SRTrainingConfig:
    """How to train the SR model — affects optimizer setup and weight init.

    Args:
        layer_lrs: Absolute per-``Conv2d`` learning rates (one entry per
            ``Conv2d`` in the model, in module-traversal order).  When
            ``None`` (default), training uses the optimizer's base ``lr``
            uniformly across all parameters.  Only valid for architectures
            where every trainable parameter lives in a ``Conv2d`` (no
            BatchNorm / PReLU); :meth:`SRLightning.configure_optimizers`
            raises ``ValueError`` otherwise.

        example_input_shape: Shape of a single input sample *excluding* the
            batch dimension (e.g. ``(1, 33, 33)`` for a 33×33 Y-channel patch).
            When provided, ``self.example_input_array`` is set so the
            TensorBoard logger can capture the model graph.

        init_strategy: ``'paper'`` triggers a paper-faithful weight init via
            :meth:`SRModel.reset_parameters` in :class:`SRLightning`'s constructor;
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
    init_strategy: Literal['default', 'paper'] = 'default'
    init_mean: float = 0.0
    init_std: float = 0.01


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
    psnr_channels: list[str] = field(default_factory=lambda: ['RGB'])
    separate_psnr: bool = False

    def __post_init__(self) -> None:
        """Validate ``psnr_channels`` at construction.

        Raises:
            ValueError: If any entry is not a supported colorspace
                (``'RGB'`` or ``'YCbCr'``).
        """
        valid = ('RGB', 'YCbCr')
        invalid = [c for c in self.psnr_channels if c not in valid]
        if invalid:
            raise ValueError(
                f"SREvalConfig.psnr_channels entries must be one of "
                f"{list(valid)}; got unsupported {invalid}. Fix "
                f"model.eval_config.init_args.psnr_channels in your YAML."
            )
