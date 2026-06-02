"""SRResNet-paper-faithful training and evaluation defaults.

Subclassing :class:`~sisr.training.config.SRTrainingConfig` /
:class:`~sisr.training.config.SREvalConfig` lets the SRResNet recipe live
alongside the model architecture, so an SRResNet experiment YAML only has
to point at these classes via ``class_path`` to inherit defaults.

Reference: Photo-Realistic Single Image Super-Resolution Using a
Generative Adversarial Network (https://arxiv.org/pdf/1609.04802),
Section 3.2 (SRResNet baseline).
"""
from dataclasses import dataclass, field
from typing import Literal

from sisr.training.config import SREvalConfig, SRTrainingConfig


@dataclass
class SRResNetTrainingConfig(SRTrainingConfig):
    """SRResNet-paper-faithful training defaults.

    The paper trains on full RGB (selected at the YAML layer by pairing
    with :class:`~sisr.processors.RGBProcessor`) using Adam (lr 1e-4) and
    no per-layer LRs.

    Weight initialization in the paper is implicit (Kaiming-style for PReLU
    activations). ``init_strategy='paper'`` is reserved for a future PR
    that wires this up via :meth:`SRResNet.reset_parameters`. Today the
    field defaults to ``'default'`` so SRResNet ships with PyTorch's
    built-in init; flipping to ``'paper'`` is currently a no-op because
    :meth:`SRModel.reset_parameters` (the inherited base) does nothing.
    The field exists now so a future PR can add the implementation without
    a YAML schema change.
    """

    init_strategy: Literal['default', 'paper'] = 'default'


@dataclass
class SRResNetEvalConfig(SREvalConfig):
    """SRResNet-paper-faithful eval defaults.

    Reports PSNR on RGB and YCbCr (literature typically quotes Y-channel
    for SRResNet too); excludes the outer ``scale=4`` pixels before
    computing PSNR / SSIM, matching the standard SR-evaluation convention.

    Args:
        crop_border: Overrides the base default to ``4`` (outer pixels
            excluded before PSNR / SSIM at the standard ``x4`` scale).
        psnr_channels: Overrides the base default to
            ``['RGB', 'YCbCr']`` (the literature reports both).
    """

    crop_border: int = 4
    psnr_channels: list[str] = field(default_factory=lambda: ['RGB', 'YCbCr'])
