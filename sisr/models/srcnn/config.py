"""SRCNN-paper-faithful training and eval defaults.

Subclassing :class:`SRTrainingConfig` / :class:`SREvalConfig` from
:mod:`sisr.training.config` lets the SRCNN-paper recipe live alongside the
model architecture, so an SRCNN experiment YAML only has to point at these
classes via ``class_path`` to inherit defaults.

Reference: Image Super-Resolution Using Deep Convolutional Networks
(https://arxiv.org/pdf/1501.00092).
"""
from dataclasses import dataclass, field
from typing import Literal

from sisr.training.config import SREvalConfig, SRTrainingConfig


@dataclass
class SRCNNTrainingConfig(SRTrainingConfig):
    """SRCNN-paper-faithful training defaults.

    The paper trains on the Y channel of YCbCr only (selected at the YAML
    layer by pairing with :class:`~sisr.processors.YChannelProcessor`) and
    uses a per-layer learning rate of ``1e-4`` for the feature-extraction
    and non-linear-mapping layers and ``1e-5`` for the reconstruction layer
    — i.e. the last layer learns 10× slower. Weight initialization follows
    the paper's Gaussian schedule (``N(0, 0.01)`` with zero biases); set
    ``init_strategy='default'`` to fall back to PyTorch's built-in init.
    Override any field in YAML to deviate.

    Args:
        layer_lrs: Per-``Conv2d`` LRs ``[1e-4, 1e-4, 1e-5]`` matching the
            paper's recipe. Set to ``None`` to disable per-layer LRs.
        init_strategy: ``'paper'`` (default) triggers the Gaussian
            init via :meth:`SRCNN.reset_parameters` in
            :class:`~sisr.training.SRLightning`'s constructor;
            ``'default'`` skips it and uses PyTorch's defaults.
        init_mean / init_std: Gaussian parameters used by
            ``init_strategy='paper'``; inherited from
            :class:`~sisr.training.config.SRTrainingConfig` with defaults
            ``0.0`` / ``0.01`` matching the SRCNN paper. Override in YAML
            to deviate.
    """

    layer_lrs: list[float] | None = field(
        default_factory=lambda: [1.0e-4, 1.0e-4, 1.0e-5]
    )
    init_strategy: Literal['default', 'paper'] = 'paper'


@dataclass
class SRCNNEvalConfig(SREvalConfig):
    """SRCNN-paper-faithful eval defaults.

    Reports PSNR on both RGB and the YCbCr triplet (the literature
    usually quotes Y-channel PSNR for SRCNN), and excludes the outer
    ``scale=3`` pixels per the standard SR-evaluation convention.

    Args:
        crop_border: Overrides the base default to ``3`` (outer pixels
            excluded before PSNR / SSIM at the standard ``x3`` scale).
        psnr_channels: Overrides the base default to
            ``['RGB', 'YCbCr']`` (the literature reports both).
    """

    crop_border: int = 3
    psnr_channels: list[str] = field(default_factory=lambda: ['RGB', 'YCbCr'])
