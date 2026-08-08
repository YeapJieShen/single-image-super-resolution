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
        init_from: Path to a bare-weights ``.pt`` whose generator weights
            initialise this run's generator. The paper scopes the MSE-init
            trick to "when training the actual GAN", so a paper-faithful SRGAN
            run starts from an MSE-trained SRResNet. Optional: unset trains
            from scratch, which is what tests and smoke runs use.

        adversarial_weight: Weight on the adversarial term of the generator's
            loss. Defaults to ``1e-3``, the paper's value. The content term is
            whatever ``criterion`` the module was given, so the total is
            ``content + adversarial_weight * adversarial``.

        d_steps_per_g_step: Goodfellow's ``k`` — discriminator updates per
            generator update. Defaults to ``1``, which is Ledig's 1:1
            alternation.

    Raises:
        ValueError: If ``d_steps_per_g_step`` is below 1, or ``cuda_graph`` is
            set.
    """

    init_from: str | None = None
    adversarial_weight: float = 1e-3
    d_steps_per_g_step: int = 1

    def __post_init__(self) -> None:
        """Reject settings this training mode cannot honour."""
        super().__post_init__()
        if self.d_steps_per_g_step < 1:
            raise ValueError(f"d_steps_per_g_step must be >= 1; got {self.d_steps_per_g_step}.")
        if self.cuda_graph:
            raise ValueError(
                "training_config.cuda_graph=True is not supported for adversarial "
                "training: CUDAGraphStep captures one {zero_grad, forward, loss, "
                "backward} around one optimizer, and this mode runs two optimizers "
                "alternately under manual optimization. Set cuda_graph to false."
            )


@dataclass
class SRGANEvalConfig(SRResNetEvalConfig):
    """SRGAN eval defaults — SRResNet's scoring, plus perceptual metrics.

    Inherits ``crop_border=4``, ``psnr_channels=['RGB', 'Y']`` and
    ``ssim_impl='daala'``, so an SRGAN number stays comparable to the SRResNet
    baseline computed the same way.

    Args:
        perceptual_metrics: Overrides the base default to ``['lpips', 'dists']``.
            An adversarial objective makes PSNR and SSIM **worse by design**, so
            without these an SRGAN run has no metric that tracks what it is
            optimising. ``'lpips'`` requires the ``[perceptual]`` extra.
    """

    perceptual_metrics: list[str] = field(default_factory=lambda: ["lpips", "dists"])
