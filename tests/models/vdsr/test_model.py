import pytest
import torch

from sisr.models.base import SRModelOutput
from sisr.models.vdsr import VDSR


def _model(depth=4, features=6, c=1):
    torch.manual_seed(0)
    return VDSR(in_out_channels=c, depth=depth, features=features)


def _convs(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]


@pytest.mark.parametrize("c", [1, 3])
def test_forward_preserves_shape_and_returns_a_bare_tensor(c):
    x = torch.rand(2, c, 9, 11)

    out = _model(c=c)(x)

    assert out.shape == x.shape and not isinstance(out, SRModelOutput)
    assert VDSR.input_contract == "pre_upsampled" and VDSR.multi_scale is False


def test_the_global_residual_adds_the_input_to_the_prediction():
    """With the last conv zeroed the residual is 0, so the output must be the input."""
    model = _model()
    last = _convs(model)[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    x = torch.rand(1, 1, 8, 8)

    assert torch.equal(model(x), x)
    with torch.no_grad():
        last.bias.fill_(0.25)
    assert torch.allclose(model(x), x + 0.25)


def test_depth_sets_the_conv_count_and_a_relu_follows_all_but_the_last():
    for depth in (2, 5, 20):
        model = _model(depth=depth, features=2)
        assert len(_convs(model)) == depth
        kinds = [type(m) for m in model.body]
        assert kinds[-1] is torch.nn.Conv2d and kinds.count(torch.nn.ReLU) == depth - 1
    n = lambda d: sum(p.numel() for p in _model(depth=d).parameters())  # noqa: E731
    assert n(6) > n(4)


def test_default_is_the_papers_20_layer_64_wide_net():
    model = VDSR()
    convs = _convs(model)

    assert len(convs) == 20 and model.hparams["features"] == 64
    assert convs[0].in_channels == 1 and convs[0].out_channels == 64
    assert convs[-1].out_channels == 1 and convs[10].weight.shape == (64, 64, 3, 3)


@pytest.mark.parametrize(
    "kwargs",
    [{"depth": 1}, {"depth": 0}, {"kernel_size": 4}, {"features": True}, {"in_out_channels": 0}],
)
def test_invalid_sizes_raise(kwargs):
    with pytest.raises(ValueError):
        VDSR(**kwargs)


def test_he_init_scales_each_conv_by_its_fan_in_and_zeroes_biases():
    model = VDSR(depth=3, features=128)
    with torch.no_grad():
        for m in _convs(model):
            m.bias.fill_(1.0)

    model.reset_parameters()

    for m in _convs(model):
        fan_in = m.in_channels * 9
        assert m.weight.std().item() == pytest.approx((2 / fan_in) ** 0.5, rel=0.1)
        assert torch.equal(m.bias, torch.zeros_like(m.bias))


def test_variant_tag_and_hparams():
    model = VDSR(depth=20)

    assert model.variant_tag == "d20"
    assert model.hparams == {"in_out_channels": 1, "depth": 20, "features": 64, "kernel_size": 3}
