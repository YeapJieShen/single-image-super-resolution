import pytest
import torch

from sisr.models.srresnet.model import SRResidualBlock, SRResNet, SRUpsampleBlock


def test_package_reexports_public_symbols():
    """SRResNet et al. are importable from the package, not just .model."""
    from sisr.models.srresnet import SRResNet as PkgSRResNet
    from sisr.models.srresnet import SRResidualBlock as PkgBlock
    from sisr.models.srresnet import SRUpsampleBlock as PkgUpsample
    assert PkgSRResNet is SRResNet
    assert PkgBlock is SRResidualBlock
    assert PkgUpsample is SRUpsampleBlock


def test_hparams_exposes_architecture():
    model = SRResNet(scale=4, hidden_channel=32, num_residual_blocks=2)
    h = model.hparams
    assert h["scale"] == 4
    assert h["hidden_channel"] == 32
    assert h["num_residual_blocks"] == 2
    assert h["in_out_channels"] == 3


def test_check_scale_one_is_valid():
    """scale=1 is a power of 2 (2**0) — log2(1)=0 upsample blocks, identity size."""
    model = SRResNet(scale=1, num_residual_blocks=1)
    out = model(torch.zeros(1, 3, 8, 8))
    assert out.shape == (1, 3, 8, 8)


def test_forward_x2():
    model = SRResNet(scale=2, num_residual_blocks=2)
    x = torch.zeros(2, 3, 16, 16)
    out = model(x)
    assert out.shape == (2, 3, 32, 32)


def test_forward_x4():
    model = SRResNet(scale=4, num_residual_blocks=2)
    x = torch.zeros(2, 3, 16, 16)
    out = model(x)
    assert out.shape == (2, 3, 64, 64)


def test_forward_x8():
    """log2(8) = 3 upsample blocks should produce 8x output."""
    model = SRResNet(scale=8, num_residual_blocks=1)
    x = torch.zeros(1, 3, 8, 8)
    out = model(x)
    assert out.shape == (1, 3, 64, 64)


def test_check_scale_non_power_of_two_raises():
    with pytest.raises(ValueError):
        SRResNet(scale=3)


def test_check_scale_six_raises():
    """6 = 2*3 — not a power of 2."""
    with pytest.raises(ValueError):
        SRResNet(scale=6)


def test_check_scale_negative_raises():
    with pytest.raises(ValueError):
        SRResNet(scale=-2)


def test_check_scale_non_int_raises():
    with pytest.raises(ValueError):
        SRResNet(scale=2.0)


def test_residual_block_preserves_shape():
    block = SRResidualBlock(channels=16, kernel_size=3)
    x = torch.zeros(2, 16, 8, 8)
    out = block(x)
    assert out.shape == x.shape


def test_residual_block_is_identity_when_branch_zeroed():
    """block(x) = x + branch(x), where branch = conv->BN->PReLU->conv->BN.
    Zero the SECOND conv's weight+bias so its output is 0; in eval mode the
    trailing BatchNorm maps 0 -> 0 (running_mean=0, bias=0), collapsing the whole
    residual branch to 0. The block must then reproduce its (random, non-zero)
    input exactly. The old test fed zeros and only asserted the shape, so a
    dropped/renamed skip connection would have passed unnoticed."""
    torch.manual_seed(0)
    block = SRResidualBlock(channels=16, kernel_size=3).eval()
    with torch.no_grad():
        block.block2[0].weight.zero_()   # block2 = Sequential(Conv2d, BatchNorm2d)
        block.block2[0].bias.zero_()
    x = torch.rand(1, 16, 4, 4)           # random, non-zero: identity must survive
    with torch.no_grad():
        out = block(x)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=0)


def test_upsample_block_doubles_spatial():
    block = SRUpsampleBlock(channels=16, scale=2)
    x = torch.zeros(2, 16, 4, 4)
    out = block(x)
    assert out.shape == (2, 16, 8, 8)


def test_upsample_block_quadruples_with_scale_4():
    block = SRUpsampleBlock(channels=16, scale=4)
    x = torch.zeros(1, 16, 4, 4)
    out = block(x)
    assert out.shape == (1, 16, 16, 16)
