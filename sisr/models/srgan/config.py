"""SRGAN-paper-faithful training and evaluation defaults.

Reference: Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network (https://arxiv.org/pdf/1609.04802), Section 3.2.
"""

from dataclasses import dataclass, field

from sisr.models.srresnet.config import SRResNetEvalConfig
from sisr.training.config import AdversarialTrainingConfig


@dataclass
class SRGANTrainingConfig(AdversarialTrainingConfig):
    """SRGAN training defaults.

    The generator is SRResNet, so ``scale=4`` is this class's own default.
    Everything adversarial-specific (``init_from``, ``adversarial_weight``,
    ``d_steps_per_g_step``, and their validation) is inherited unchanged from
    :class:`~sisr.training.config.AdversarialTrainingConfig`.
    """

    scale: int = 4


@dataclass
class SRGANEvalConfig(SRResNetEvalConfig):
    """SRGAN eval defaults — SRResNet's scoring, plus perceptual metrics.

    Inherits SRResNet's border, channels and ``ssim_impl='daala'``, so an SRGAN
    number stays comparable to the baseline computed the same way.

    Args:
        perceptual_metrics: ``['lpips', 'dists']``. An adversarial objective
            makes PSNR and SSIM **worse by design**, so without these a run has
            no metric tracking what it optimises. ``'lpips'`` needs the
            ``[perceptual]`` extra.
    """

    perceptual_metrics: list[str] = field(default_factory=lambda: ["lpips", "dists"])
