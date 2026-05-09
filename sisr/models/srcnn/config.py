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

    The paper trains on the Y channel of YCbCr only and uses a per-layer
    learning rate of ``1e-4`` for the feature-extraction and non-linear-
    mapping layers and ``1e-5`` for the reconstruction layer — i.e. the
    last layer learns 10× slower.  Override any field in YAML to deviate
    (e.g. ``layer_lrs: null`` for uniform LR, or ``model_colorspace: RGB``
    for full-RGB training).
    """

    model_colorspace: Literal['RGB', 'Y', 'YCbCr'] = 'Y'
    layer_lrs: list[float] | None = field(
        default_factory=lambda: [1.0e-4, 1.0e-4, 1.0e-5]
    )


@dataclass
class SRCNNEvalConfig(SREvalConfig):
    """SRCNN-paper-faithful eval defaults.

    Reports PSNR on both RGB and the YCbCr triplet (the literature usually
    quotes Y-channel PSNR for SRCNN), and excludes the outer ``scale=3``
    pixels per the standard SR-evaluation convention.
    """

    crop_border: int = 3
    psnr_channels: list[str] = field(default_factory=lambda: ['RGB', 'YCbCr'])
