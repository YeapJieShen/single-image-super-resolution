"""Per-architecture training and evaluation configuration dataclasses.

Split into two classes by lifecycle:

* :class:`SRTrainingConfig` controls behaviour during ``cli fit`` — model
  input/output colorspace, per-Conv2d learning rates, and the example input
  shape used to log the model graph to TensorBoard.
* :class:`SREvalConfig` controls validation/test metric computation —
  boundary-pixel exclusion (``crop_border``) and which colorspaces PSNR is
  reported in.

Per-architecture defaults live in subclasses next to the model code (e.g.
``sisr.models.srcnn.SRCNNTrainingConfig``); a YAML user picks them via
``class_path`` on ``model.training_config`` / ``model.eval_config`` and
overrides individual fields with ``init_args``.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SRTrainingConfig:
    """How to train the SR model — affects forward pass, loss, and optimizer setup.

    Args:
        model_colorspace: Colorspace the inner model trains on.

            * ``'RGB'`` (default) — model receives and emits RGB directly.
            * ``'Y'`` — Y channel of LR YCbCr is fed to the model; the SR
              output is stitched with bicubic Cb/Cr taken from the LR image
              and converted back to RGB before metrics are computed.
            * ``'YCbCr'`` — model trains on full YCbCr; output converted
              back to RGB for metrics.

            The dataset always serves RGB; conversion happens inside
            :class:`SRLightning` via :mod:`sisr.utils`.

        layer_lrs: Absolute per-``Conv2d`` learning rates (one entry per
            ``Conv2d`` in the model, in module-traversal order).  When
            ``None`` (default), training uses the optimizer's base ``lr``
            uniformly across all parameters.  Only valid for architectures
            where every trainable parameter lives in a ``Conv2d`` (no
            BatchNorm / PReLU); :meth:`SRLightning.configure_optimizers`
            raises ``ValueError`` otherwise.

        example_input_shape: Shape of a single input sample *excluding* the
            batch dimension (e.g. ``(3, 33, 33)`` for a 33×33 RGB patch).
            When provided, ``self.example_input_array`` is set so the
            TensorBoard logger can capture the model graph.
    """

    model_colorspace: Literal['RGB', 'Y', 'YCbCr'] = 'RGB'
    layer_lrs: list[float] | None = None
    example_input_shape: tuple[int, ...] | None = None


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
