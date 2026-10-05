import pytest
import torch

from sisr.models.esrgan import RRDBNet, VGGStyleDiscriminator
from sisr.models.esrgan.model import _RRDB, _DenseBlock


def _net(**kw):
    torch.manual_seed(0)
    return RRDBNet(**{"features": 8, "growth": 4, "num_blocks": 2} | kw)


@pytest.mark.parametrize("scale", [2, 4])
def test_forward_upscales_by_scale(scale):
    out = _net(scale=scale)(torch.rand(2, 3, 6, 7))

    assert out.shape == (2, 3, 6 * scale, 7 * scale)
    assert RRDBNet.input_contract == "native_lr" and RRDBNet.multi_scale is False


def test_there_is_no_batch_norm_anywhere():
    """ESRGAN's first change to SRGAN's generator."""
    assert not [m for m in _net().modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def test_upsampling_is_nearest_neighbour_not_a_pixel_shuffle():
    net = _net()
    assert not [m for m in net.modules() if isinstance(m, torch.nn.PixelShuffle)]
    for conv in net.up:  # an all-ones identity-ish conv exposes the resize's blockiness
        torch.nn.init.zeros_(conv.weight)
        torch.nn.init.zeros_(conv.bias)
        with torch.no_grad():
            conv.weight[:, :, 1, 1] = torch.eye(8)

    probe = []
    net.up[0].register_forward_hook(lambda m, i, o: probe.append(i[0]))
    net(torch.rand(1, 3, 5, 5))

    x = probe[0]  # the resized features entering the first upsampling conv
    assert x.shape[-1] == 10
    assert torch.equal(x[..., 0::2, :], x[..., 1::2, :]) and torch.equal(x[..., 0::2], x[..., 1::2])


def test_the_residual_scale_multiplies_both_levels():
    """``x + beta * f(x)`` in a dense block and again in an RRDB, checked on stubs."""
    f = 4
    block = _DenseBlock(f, 2, beta=0.5)
    for p in block.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        block.conv5.bias.fill_(2.0)  # x5 = 2 everywhere
    x = torch.rand(1, f, 5, 5)
    assert torch.allclose(block(x), x + 0.5 * 2.0)

    rrdb = _RRDB(f, 2, beta=0.5)
    for p in rrdb.parameters():
        torch.nn.init.zeros_(p)
    for rdb in (rrdb.rdb1, rrdb.rdb2, rrdb.rdb3):
        with torch.no_grad():
            rdb.conv5.bias.fill_(2.0)
    # each dense block adds 0.5*2; three in a row add 3.0; the RRDB adds 0.5 * (x + 3) onto x...
    # ... but rdb(x) = x + 1 so f(x) = x + 3 and the RRDB gives x + 0.5 * (x + 3).
    assert torch.allclose(rrdb(x), x + 0.5 * (x + 3.0))


def test_dense_connections_widen_each_conv_input():
    block = _DenseBlock(8, 4, beta=0.2)

    assert [
        c.in_channels for c in (block.conv1, block.conv2, block.conv3, block.conv4, block.conv5)
    ] == [8, 12, 16, 20, 24]
    assert block.conv5.out_channels == 8


def test_default_is_the_papers_23_block_net_with_the_released_parameter_count():
    net = RRDBNet()

    assert len(net.trunk) == 23 and net.variant_tag == "23B32G"
    assert sum(p.numel() for p in net.parameters()) == 16_697_987


@pytest.mark.parametrize(
    "kwargs",
    [{"scale": 3}, {"scale": 1}, {"num_blocks": 0}, {"residual_scale": 0.0}, {"growth": True}],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        RRDBNet(**kwargs)


def test_paper_init_is_a_tenth_of_he_with_zero_bias():
    net = RRDBNet(num_blocks=1, features=64, growth=32)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.bias.fill_(1.0)

    net.reset_parameters()

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_in = m.in_channels * 9
            assert m.weight.std().item() == pytest.approx(0.1 * (2 / fan_in) ** 0.5, rel=0.1)
            assert torch.equal(m.bias, torch.zeros_like(m.bias))


def test_hparams_carry_the_scale_srgan_lightning_checks():
    assert _net(scale=2).hparams["scale"] == 2


# --- the discriminator ------------------------------------------------------------------


def test_discriminator_emits_one_logit_per_image_and_declares_its_input_size():
    d = VGGStyleDiscriminator(hr_input_size=32, base_channels=4, dense_features=8)

    out = d(torch.rand(3, 3, 32, 32))

    assert out.shape == (3, 1)
    assert d.input_size == 32 and d.in_channels == 3 and d.variant_tag == "32"


def test_discriminator_default_is_the_128_patch_vgg_with_five_downsamples():
    d = VGGStyleDiscriminator()

    strided = [m for m in d.features if isinstance(m, torch.nn.Conv2d) and m.stride == (2, 2)]
    assert len(strided) == 5 and all(m.kernel_size == (4, 4) for m in strided)
    assert d.classifier[0].in_features == 512 * 4 * 4 and d.classifier[-1].out_features == 1


@pytest.mark.parametrize("size", [0, 16, 100])
def test_discriminator_rejects_a_size_the_dense_head_cannot_take(size):
    with pytest.raises(ValueError, match="multiple of 32"):
        VGGStyleDiscriminator(hr_input_size=size)
