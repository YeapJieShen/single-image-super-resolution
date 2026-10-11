"""Adversarial objectives: SRGAN's non-saturating one and ESRGAN's relativistic-average one."""

import torch


class AdversarialLoss(torch.nn.Module):
    """Non-saturating GAN objective over discriminator **logits**.

    Deliberately **not** an :class:`~sisr.losses.base.SRLoss`: that contract is
    ``forward(pred, target)`` and the generator term ``-log D(G(I_LR))`` has no
    target, so fitting it would put a dummy argument in the signature the whole
    loss library depends on. Standing apart also makes it testable without a
    training loop and swappable from YAML. :meth:`generator_loss` takes both
    discriminator logit tensors, so a relativistic subclass (e.g. the
    relativistic-average objective ESRGAN uses) reads ``logits_real`` straight
    off its own signature instead of reaching outside it for a side-channel.
    This class's own non-saturating form ignores ``logits_real`` entirely.

    Takes **logits**, not probabilities. Parameter-free, so it adds nothing to
    any ``state_dict``.
    """

    def generator_loss(self, logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
        """Generator's adversarial term: make the discriminator call fakes real.

        The **non-saturating** form (BCE against a *real* target), not
        ``+log(1 - D(G(x)))``, whose gradient vanishes exactly when the
        discriminator is winning and the generator most needs signal.
        **Ignores** ``logits_real`` -- this loss is not relativistic; the
        parameter exists so every subclass shares one signature, including one
        that needs it.

        Args:
            logits_real: Discriminator logits for the real HR batch, ``(B, 1)``.
                Detached and computed from the discriminator **before** its update
                (one alternation behind ``logits_fake``), so a relativistic
                subclass sees the critic's state from the discriminator step.
                Unused by this non-saturating form.
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


class RelativisticAverageLoss(AdversarialLoss):
    """The relativistic-average GAN objective of ESRGAN (Wang et al. 2018, eq. 1-2).

    The critic is asked whether a real image is *more realistic than the average fake*, and a
    fake *less realistic than the average real*: ``D_Ra(x_r, x_f) = sigmoid(C(x_r) -
    mean(C(x_f)))``, means taken over the batch. With ``C`` the logits::

        L_D = BCE(C_r - mean(C_f), 1) + BCE(C_f - mean(C_r), 0)
        L_G = BCE(C_r - mean(C_f), 0) + BCE(C_f - mean(C_r), 1)

    These are the paper's equations as written, a **sum** of the two terms. The authors'
    released training code averages them (a factor of one half) before weighting; at the
    paper's ``lambda = 5e-3`` that is an effective 2.5e-3 on this sum, so match it by halving
    ``adversarial_weight`` if reproducing their numbers rather than the equations.

    Unlike the non-saturating form, the generator term **reads ``logits_real``**: the generator
    is trained against the real data's critic scores too. They arrive detached, from the
    discriminator before its update (see :meth:`AdversarialLoss.generator_loss`).
    """

    def generator_loss(self, logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
        """``BCE(C_r - mean(C_f), 0) + BCE(C_f - mean(C_r), 1)``.

        Args:
            logits_real: Critic logits for the real batch, ``(B, 1)``, detached.
            logits_fake: Critic logits for generated images, ``(B, 1)``.

        Returns:
            0-dim tensor.
        """
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        real_rel = logits_real - logits_fake.mean()
        fake_rel = logits_fake - logits_real.mean()
        return bce(real_rel, torch.zeros_like(real_rel)) + bce(fake_rel, torch.ones_like(fake_rel))

    def discriminator_loss(
        self, logits_real: torch.Tensor, logits_fake: torch.Tensor
    ) -> torch.Tensor:
        """``BCE(C_r - mean(C_f), 1) + BCE(C_f - mean(C_r), 0)``.

        Args:
            logits_real: Critic logits for real images, ``(B, 1)``.
            logits_fake: Critic logits for generated images, ``(B, 1)``, detached.

        Returns:
            0-dim tensor.
        """
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        real_rel = logits_real - logits_fake.mean()
        fake_rel = logits_fake - logits_real.mean()
        return bce(real_rel, torch.ones_like(real_rel)) + bce(fake_rel, torch.zeros_like(fake_rel))
