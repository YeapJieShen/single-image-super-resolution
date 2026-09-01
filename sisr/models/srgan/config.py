"""SRGAN-paper-faithful training and evaluation defaults.

Reference: Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network (https://arxiv.org/pdf/1609.04802), Section 3.2.
"""

from dataclasses import dataclass, field

from sisr.models.srresnet.config import SRResNetEvalConfig, SRResNetTrainingConfig


@dataclass
class SRGANTrainingConfig(SRResNetTrainingConfig):
    """SRGAN training defaults. The generator is SRResNet, so ``scale=4`` is inherited.

    Args:
        init_from: Bare-weights ``.safetensors`` to initialise the generator
            from. A paper-faithful SRGAN run starts from an MSE-trained
            SRResNet -- the paper scopes that to "when training the actual
            GAN". Unset trains from scratch, which tests and smoke runs use.

        adversarial_weight: Weight on the generator's adversarial term;
            ``1e-3`` is the paper's. Total is
            ``content + adversarial_weight * adversarial``, where content is
            whatever ``criterion`` the module was given.

        d_steps_per_g_step: Goodfellow's ``k``. ``1`` is Ledig's alternation.

    Raises:
        ValueError: If ``d_steps_per_g_step`` is below 1.
    """

    init_from: str | None = None
    adversarial_weight: float = 1e-3
    d_steps_per_g_step: int = 1

    def __post_init__(self) -> None:
        """Reject settings this training mode cannot honour."""
        if self.d_steps_per_g_step < 1:
            raise ValueError(f"d_steps_per_g_step must be >= 1; got {self.d_steps_per_g_step}.")


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
