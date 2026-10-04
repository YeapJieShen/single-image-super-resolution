import pytest
import torch

from sisr.models.base import SRModelOutput, unwrap_primary
from sisr.models.drcn import DRCN


def _model(c=1, f=8, d=3) -> DRCN:
    torch.manual_seed(0)
    return DRCN(in_out_channels=c, features=f, recursions=d)


@pytest.mark.parametrize("c", [1, 3])
def test_forward_returns_a_record_with_stacked_recursions(c):
    x = torch.rand(2, c, 9, 11)

    out = _model(c=c)(x)

    assert isinstance(out, SRModelOutput)
    assert out.primary.shape == x.shape
    assert out.extras["recursions"].shape == (2, 3, c, 9, 11)
    assert unwrap_primary(out) is out.primary
    assert DRCN.input_contract == "pre_upsampled"


def test_the_recursive_conv_is_shared_so_parameters_do_not_grow_with_depth():
    def n_params(d):
        return sum(p.numel() for p in _model(d=d).parameters()) - d

    assert n_params(2) == n_params(16)
    convs = [m for m in _model(d=5).modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 5  # embed x2, one inference, recon x2


def test_the_input_skip_feeds_every_reconstruction():
    model = _model()
    for conv in (model.recon2,):
        torch.nn.init.zeros_(conv.weight)
        torch.nn.init.zeros_(conv.bias)
    x = torch.rand(1, 1, 6, 6)

    out = model(x)

    for d in range(3):
        assert torch.equal(out.extras["recursions"][:, d], x)
    assert torch.allclose(out.primary, x * model.ensemble.sum())


def test_ensemble_weights_are_learned_and_form_the_output():
    model = _model()
    with torch.no_grad():
        model.ensemble.copy_(torch.tensor([0.2, 0.5, -0.1]))
    x = torch.rand(1, 1, 6, 6)

    out = model(x)
    manual = (out.extras["recursions"] * model.ensemble.view(1, -1, 1, 1, 1)).sum(dim=1)
    out.primary.sum().backward()

    assert model.ensemble.requires_grad
    assert torch.allclose(out.primary, manual)
    assert model.ensemble.grad is not None and model.ensemble.grad.abs().sum() > 0


def test_ensemble_is_initialised_to_the_plain_average():
    assert torch.allclose(_model(d=4).ensemble, torch.full((4,), 0.25))


def test_paper_init_is_identity_on_the_recursive_conv_only():
    model = _model()
    with torch.no_grad():
        model.ensemble.fill_(9.0)

    model.reset_parameters()

    w = model.infer.weight
    assert torch.equal(w[3, 3, 1, 1], torch.tensor(1.0))
    assert w.sum().item() == pytest.approx(w.shape[0])
    assert model.embed[0].weight.abs().sum() > 0 and model.recon2.weight.abs().sum() > 0
    assert all(
        torch.equal(m.bias, torch.zeros_like(m.bias))
        for m in model.modules()
        if isinstance(m, torch.nn.Conv2d)
    )
    assert torch.allclose(model.ensemble, torch.full((3,), 1 / 3))


def test_variant_tag_and_hparams():
    model = DRCN(recursions=16, features=4)
    assert model.variant_tag == "r16"
    assert model.hparams == {
        "in_out_channels": 1,
        "features": 4,
        "recursions": 16,
        "kernel_size": 3,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"features": 0},
        {"recursions": 0},
        {"in_out_channels": -1},
        {"kernel_size": 4},
        {"recursions": 2.5},
    ],
)
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        DRCN(**kwargs)
