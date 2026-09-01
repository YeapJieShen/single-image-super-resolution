"""The SRGAN adversarial objective (Ledig et al. §2.2)."""

import torch


class AdversarialLoss(torch.nn.Module):
    """Non-saturating GAN objective over discriminator **logits**.

    Deliberately **not** an :class:`~sisr.losses.base.SRLoss`: that contract is
    ``forward(pred, target)`` and the generator term ``-log D(G(I_LR))`` has no
    target, so fitting it would put a dummy argument in the signature the whole
    loss library depends on. Standing apart also makes it testable without a
    training loop and swappable from YAML (LSGAN, relativistic).

    Takes **logits**, not probabilities. Parameter-free, so it adds nothing to
    any ``state_dict``.
    """

    def generator_loss(self, logits_fake: torch.Tensor) -> torch.Tensor:
        """Generator's adversarial term: make the discriminator call fakes real.

        The **non-saturating** form (BCE against a *real* target), not
        ``+log(1 - D(G(x)))``, whose gradient vanishes exactly when the
        discriminator is winning and the generator most needs signal.

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
