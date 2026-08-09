"""The SRGAN adversarial objective (Ledig et al. §2.2)."""

import torch


class AdversarialLoss(torch.nn.Module):
    """Non-saturating GAN objective over discriminator **logits**.

    Deliberately **not** an :class:`~sisr.losses.base.SRLoss`: that contract is
    ``forward(pred, target)`` in model space, and the generator's term
    ``-log D(G(I_LR))`` has no target at all. Bending it to fit would put a
    dummy argument in the signature the whole loss library depends on. Being a
    separate class also makes the objective testable without a training loop
    and swappable from YAML (LSGAN, relativistic) without touching the module.

    Takes logits, not probabilities — see
    :class:`~sisr.models.srgan.SRDiscriminator`. Parameter-free, so it adds
    nothing to any ``state_dict``.
    """

    def generator_loss(self, logits_fake: torch.Tensor) -> torch.Tensor:
        """Generator's adversarial term: make the discriminator call fakes real.

        The **non-saturating** form (``-log D(G(x))``, i.e. BCE against a
        *real* target) rather than ``+log(1 - D(G(x)))`` — the latter's gradient
        vanishes exactly when the discriminator is winning, which is when the
        generator most needs signal.

        Args:
            logits_fake: Discriminator logits for generated images, ``(B, 1)``.

        Returns:
            0-dim tensor.
        """
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits_fake, torch.ones_like(logits_fake)
        )

    def discriminator_loss(
        self, logits_real: torch.Tensor, logits_fake: torch.Tensor
    ) -> torch.Tensor:
        """Discriminator's objective: real as real, fake as fake.

        Args:
            logits_real: Logits for ground-truth HR images, ``(B, 1)``.
            logits_fake: Logits for generated images, ``(B, 1)``. The caller
                passes these **detached** — the discriminator update must not
                propagate into the generator.

        Returns:
            0-dim tensor.
        """
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        return bce(logits_real, torch.ones_like(logits_real)) + bce(
            logits_fake, torch.zeros_like(logits_fake)
        )
