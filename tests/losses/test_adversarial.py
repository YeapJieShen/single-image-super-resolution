import torch
import torch.nn.functional as F

from sisr.losses import AdversarialLoss


def test_generator_loss_is_non_saturating_bce_against_real():
    loss = AdversarialLoss()
    logits = torch.tensor([[0.3], [-1.2]])
    expected = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    assert torch.allclose(loss.generator_loss(logits), expected)


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
    loss.generator_loss(logits).backward()
    assert (logits.grad < 0).all()


def test_it_holds_no_parameters():
    """A parameter-bearing criterion would land in every checkpoint's state_dict."""
    assert list(AdversarialLoss().parameters()) == []
