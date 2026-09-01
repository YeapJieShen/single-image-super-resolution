import pytest
import torch

from sisr.models.srresnet.model import SRResidualBlock, SRResNet, SRUpsampleBlock


def test_package_reexports_public_symbols():
    """SRResNet et al. are importable from the package, not just .model."""
    from sisr.models.srresnet import SRResidualBlock as PkgBlock
    from sisr.models.srresnet import SRResNet as PkgSRResNet
    from sisr.models.srresnet import SRUpsampleBlock as PkgUpsample

    assert PkgSRResNet is SRResNet
    assert PkgBlock is SRResidualBlock
    assert PkgUpsample is SRUpsampleBlock


def test_input_contract_is_native_lr():
    """SRResNet consumes true low-resolution input and upsamples internally."""
    assert SRResNet.input_contract == "native_lr"


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


def test_check_architecture_wrong_kernel_sizes_length_raises():
    """A kernel_sizes tuple that isn't length-3 must raise a clear
    ValueError at construction, not an IndexError deep in layer wiring."""
    with pytest.raises(ValueError, match="kernel_sizes"):
        SRResNet(scale=2, kernel_sizes=(9, 3))


def test_check_architecture_nonpositive_num_residual_blocks_raises():
    """num_residual_blocks <= 0 must raise a clear ValueError instead of
    silently building an empty residual Sequential."""
    with pytest.raises(ValueError, match="num_residual_blocks"):
        SRResNet(scale=2, num_residual_blocks=0)


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
        block.block2[0].weight.zero_()  # block2 = Sequential(Conv2d, BatchNorm2d)
        block.block2[0].bias.zero_()
    x = torch.rand(1, 16, 4, 4)  # random, non-zero: identity must survive
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


def test_clamp_output_default_bounds_are_the_paper_signed_range():
    """Default clamp bounds must be [-1, 1], matching RGBSignedOutputProcessor.

    A (0.0, 1.0) default would floor every negative activation at 0 — half the
    paper's tonal range, silently. Driving the tail bias negative makes the two
    candidate defaults produce different floors, so this fails decisively on the
    wrong one rather than merely reading the signature back.
    """
    model = SRResNet(scale=2, num_residual_blocks=1)
    torch.nn.init.constant_(model.tail.bias, -5.0)

    out = model(torch.zeros(1, 3, 8, 8), clamp_output=True)

    assert torch.isclose(out.min(), torch.tensor(-1.0), atol=1e-6), (
        f"floor was {out.min().item()}; a (0.0, 1.0) default floors at 0.0"
    )


def test_variant_tag_is_blocks_and_width():
    """The two knobs that move capacity, and the two a reader wants off `ls`."""
    assert SRResNet(scale=4).variant_tag == "16B64F"
    assert SRResNet(scale=4, num_residual_blocks=8, hidden_channel=32).variant_tag == "8B32F"


# ---------------------------------------------------------------------------
# padding is honoured everywhere, or rejected -- never recorded and ignored
# ---------------------------------------------------------------------------


def test_every_conv_agrees_with_the_recorded_padding_hparam():
    """`padding` threaded into head, residual blocks and tail; SRUpsampleBlock
    hardcoded 'same'. _hparams goes verbatim into checkpoint and ONNX
    provenance as model.init_args, so a file claimed padding: 1 while one of
    its convolutions was built 'same'. This project treats artifact metadata as
    what a downstream consumer trusts to know what a file means."""
    model = SRResNet(
        scale=2, hidden_channel=8, num_residual_blocks=1, kernel_sizes=(3, 3, 3), padding=1
    )
    recorded = model.hparams["padding"]
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            assert module.padding == (recorded, recorded), (
                f"{name} was built with padding={module.padding}, but the model records "
                f"padding={recorded!r}"
            )


def test_same_padding_reaches_the_upsample_conv_too():
    """The 'same' case must keep working -- it is what every shipped config uses."""
    model = SRResNet(scale=2, hidden_channel=8, num_residual_blocks=1)
    assert model.hparams["padding"] == "same"
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            assert module.padding == "same", f"{name} did not receive 'same'"


def test_padding_that_would_break_the_scale_factor_is_rejected():
    """padding=1 with the paper's 9-3-9 kernels RUNS and silently returns
    30x30 from a 24x24 input at scale 2 -- the head and tail each shrink by 6,
    so the model is no longer xscale. Nothing raised and nothing warned."""
    with pytest.raises(ValueError, match="padding"):
        SRResNet(
            scale=2, hidden_channel=8, num_residual_blocks=1, kernel_sizes=(9, 3, 9), padding=1
        )


def test_padding_that_breaks_the_residual_skip_is_rejected():
    """padding=0 made the residual block's output smaller than its input, so
    `x + block(x)` failed on shape -- a raw RuntimeError from inside forward,
    several frames from the config that caused it."""
    with pytest.raises(ValueError, match="padding"):
        SRResNet(scale=2, hidden_channel=8, num_residual_blocks=1, padding=0)


def test_shape_preserving_int_padding_is_accepted_and_scales_correctly():
    """The rule is shape-preservation, not the literal string 'same': p ==
    (k-1)//2 preserves the map for an odd kernel, so padding=1 with all-3
    kernels is legitimate and must still upscale by exactly `scale`."""
    model = SRResNet(
        scale=2, hidden_channel=8, num_residual_blocks=1, kernel_sizes=(3, 3, 3), padding=1
    )
    out = model(torch.zeros(1, 3, 24, 24))
    assert out.shape == (1, 3, 48, 48)


def test_padding_that_is_neither_same_nor_an_int_is_rejected():
    """'valid' is the other string torch accepts, and it is not shape-preserving
    for any kernel above 1. bool is excluded separately: isinstance(True, int)
    is True, and `padding: true` is not a pixel count."""
    for bad in ("valid", 1.5, True, None):
        with pytest.raises(ValueError, match="padding"):
            SRResNet(scale=2, hidden_channel=8, num_residual_blocks=1, padding=bad)


def test_even_kernel_cannot_be_shape_preserving_with_int_padding():
    """An even kernel has no p making the conv shape-preserving, so no int
    padding can be accepted for one -- (k-1)//2 would silently lose a pixel."""
    with pytest.raises(ValueError, match="padding"):
        SRResNet(
            scale=2, hidden_channel=8, num_residual_blocks=1, kernel_sizes=(4, 4, 4), padding=1
        )
