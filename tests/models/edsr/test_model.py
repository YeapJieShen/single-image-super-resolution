import pytest
import torch

from sisr.models.edsr import EDSR
from sisr.models.edsr.model import _ResBlock


def _net(scale=2, **kw):
    torch.manual_seed(0)
    return EDSR(scale=scale, **{"features": 8, "num_blocks": 2} | kw)


@pytest.mark.parametrize("scale", [2, 3, 4, 8])
def test_forward_upscales_by_scale(scale):
    out = _net(scale)(torch.rand(2, 3, 5, 6))

    assert out.shape == (2, 3, 5 * scale, 6 * scale)
    assert EDSR.input_contract == "native_lr" and EDSR.multi_scale is False


def test_upsampler_stages_follow_the_scale():
    """x2 and x3 have one conv + shuffle, x4 two (the released checkpoints' layout)."""
    shapes = {s: [c.out_channels for c in _net(s).up] for s in (2, 3, 4)}

    assert shapes == {2: [32], 3: [72], 4: [32, 32]}


def test_there_is_no_batch_norm_anywhere():
    assert not [m for m in _net().modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def test_the_residual_scale_multiplies_each_blocks_residual():
    block = _ResBlock(4, residual_scale=0.1)
    for p in block.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        block.conv2.bias.fill_(2.0)
    x = torch.rand(1, 4, 5, 5)

    assert torch.allclose(block(x), x + 0.2)


def test_mean_shift_round_trips_exactly_when_every_conv_is_zero():
    """With the body zeroed the net outputs its tail bias: ``bias / range + mean``."""
    net = _net(rgb_mean=(0.5, 0.25, 0.125))
    for p in net.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        net.tail.bias.copy_(torch.tensor([51.0, 25.5, 12.75]))

    out = net(torch.rand(1, 3, 4, 4))

    expect = torch.tensor([0.2 + 0.5, 0.1 + 0.25, 0.05 + 0.125]).view(1, 3, 1, 1)
    assert torch.allclose(out, expect.expand_as(out), atol=1e-6)


def test_the_input_is_shifted_by_the_mean_and_scaled_to_the_weights_range():
    net = _net()
    seen = []
    net.head.register_forward_hook(lambda m, i, o: seen.append(i[0]))
    x = torch.rand(1, 3, 4, 4)

    net(x)

    assert torch.allclose(seen[0], (x - net.mean) * 255.0)


def test_the_mean_is_not_a_saved_parameter():
    """It is a fixed buffer, so a checkpoint carries no mean-shift convs."""
    assert not [k for k in _net().state_dict() if "mean" in k]


def test_presets_and_variant_tags():
    baseline = _net(features=64, num_blocks=16)
    full = _net(features=256, num_blocks=32, residual_scale=0.1)

    assert (baseline.variant_tag, full.variant_tag) == ("16B64F", "32B256F")
    assert full.hparams["residual_scale"] == 0.1 and baseline.hparams["scale"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 5},
        {"scale": 1},
        {"scale": 6},
        {"num_blocks": 0},
        {"rgb_mean": (0.5,)},
        {"rgb_range": 0},
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        EDSR(**({"scale": 2} | kwargs))


def test_reset_parameters_leaves_the_default_init_alone():
    net = _net()
    before = {k: v.clone() for k, v in net.state_dict().items()}

    net.reset_parameters()

    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())
