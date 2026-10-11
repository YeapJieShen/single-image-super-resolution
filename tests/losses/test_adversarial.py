import math

import pytest
import torch
import torch.nn.functional as F

from sisr.losses import AdversarialLoss, RelativisticAverageLoss


def test_generator_loss_is_non_saturating_bce_against_real():
    loss = AdversarialLoss()
    logits = torch.tensor([[0.3], [-1.2]])
    expected = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    assert torch.allclose(loss.generator_loss(torch.zeros_like(logits), logits), expected)


def test_discriminator_loss_is_the_two_term_sum():
    loss = AdversarialLoss()
    real, fake = torch.tensor([[2.0]]), torch.tensor([[-2.0]])
    expected = F.binary_cross_entropy_with_logits(
        real, torch.ones_like(real)
    ) + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    assert torch.allclose(loss.discriminator_loss(real, fake), expected)


def test_a_confident_correct_discriminator_scores_near_zero():
    loss = AdversarialLoss()
    d = loss.discriminator_loss(torch.full((8, 1), 20.0), torch.full((8, 1), -20.0))
    assert d.item() < 1e-6


def test_generator_gradient_pushes_fake_logits_up():
    """The generator's objective is to make D call its output real."""
    loss = AdversarialLoss()
    logits = torch.zeros(4, 1, requires_grad=True)
    loss.generator_loss(torch.zeros(4, 1), logits).backward()
    assert (logits.grad < 0).all()


def test_generator_loss_ignores_logits_real_and_matches_the_old_value():
    """#274: generator_loss gained a required logits_real parameter. Guards the
    mutation 'quietly folds logits_real into the vanilla formula' — the value
    for a given logits_fake must be identical no matter what logits_real is."""
    loss = AdversarialLoss()
    logits_fake = torch.tensor([[0.3], [-1.2]])
    expected = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
    assert torch.allclose(loss.generator_loss(torch.zeros(2, 1), logits_fake), expected)
    assert torch.allclose(loss.generator_loss(torch.full((2, 1), 99.0), logits_fake), expected)


def test_it_holds_no_parameters():
    """A parameter-bearing criterion would land in every checkpoint's state_dict."""
    assert list(AdversarialLoss().parameters()) == []


# --- RelativisticAverageLoss (ESRGAN eq. 1-2) ------------------------------------------


def _softplus(x):
    return math.log1p(math.exp(x))


def _bce(x, target):  # BCE-with-logits of one scalar
    return _softplus(x) - x * target


REAL = torch.tensor([[2.0], [0.5]])
FAKE = torch.tensor([[-1.0], [1.0]])


def _ra_terms():
    """Hand computation: means are 1.25 (real) and 0.0 (fake)."""
    real_rel = [2.0 - 0.0, 0.5 - 0.0]  # C_r - mean(C_f)
    fake_rel = [-1.0 - 1.25, 1.0 - 1.25]  # C_f - mean(C_r)
    return real_rel, fake_rel


def test_relativistic_discriminator_loss_matches_the_hand_computation():
    real_rel, fake_rel = _ra_terms()
    expected = sum(_bce(x, 1) for x in real_rel) / 2 + sum(_bce(x, 0) for x in fake_rel) / 2

    got = RelativisticAverageLoss().discriminator_loss(REAL, FAKE)

    assert got.item() == pytest.approx(expected)


def test_relativistic_generator_loss_is_the_mirrored_labels():
    real_rel, fake_rel = _ra_terms()
    expected = sum(_bce(x, 0) for x in real_rel) / 2 + sum(_bce(x, 1) for x in fake_rel) / 2

    got = RelativisticAverageLoss().generator_loss(REAL, FAKE)

    assert got.item() == pytest.approx(expected)


def test_the_relativistic_generator_loss_reads_the_real_logits():
    """It is not the non-saturating loss: changing the real logits changes it."""
    ra = RelativisticAverageLoss()

    assert ra.generator_loss(REAL, FAKE).item() != pytest.approx(
        ra.generator_loss(REAL + 3, FAKE).item()
    )
    ns = AdversarialLoss()
    assert ns.generator_loss(REAL, FAKE).item() == pytest.approx(
        ns.generator_loss(REAL + 3, FAKE).item()
    )


def test_relativistic_generator_gradient_reaches_only_the_fake_logits():
    """The real logits arrive detached; the generator learns through the fake ones."""
    real = REAL.clone()
    fake = FAKE.clone().requires_grad_(True)

    RelativisticAverageLoss().generator_loss(real.detach(), fake).backward()

    assert fake.grad is not None and fake.grad.abs().sum() > 0
    assert real.grad is None


def test_relativistic_generator_loss_falls_when_fakes_look_more_real():
    ra = RelativisticAverageLoss()

    assert ra.generator_loss(REAL, FAKE + 4).item() < ra.generator_loss(REAL, FAKE).item()


def test_it_is_an_adversarial_loss_with_no_parameters():
    ra = RelativisticAverageLoss()

    assert isinstance(ra, AdversarialLoss) and not list(ra.parameters())
