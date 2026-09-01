import pytest
import torch

from sisr.models.srcnn import SRCNN


@pytest.fixture
def srcnn_valid() -> SRCNN:
    """Standard SRCNN with valid padding (kernels 9/1/5 → spatial dim shrinks by 12)."""
    return SRCNN(
        num_channels=3,
        num_filters=(64, 32),
        kernel_sizes=(9, 1, 5),
        padding=0,
    )


def test_input_contract_is_pre_upsampled():
    """SRCNN is resolution-preserving: LR arrives already bicubic-upsampled to HR size."""
    assert SRCNN.input_contract == "pre_upsampled"


def test_forward_valid_padding_shrinks_spatial(srcnn_valid: SRCNN):
    x = torch.zeros(2, 3, 33, 33)
    out = srcnn_valid(x)
    # 33 - 8 - 0 - 4 = 21 (kernel 9 -> -8, kernel 1 -> -0, kernel 5 -> -4)
    assert out.shape == (2, 3, 21, 21)


def test_forward_same_padding_preserves_spatial():
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding="same")
    x = torch.zeros(2, 3, 33, 33)
    out = model(x)
    assert out.shape == (2, 3, 33, 33)


def test_forward_y_channel():
    model = SRCNN(num_channels=1, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    x = torch.zeros(2, 1, 33, 33)
    out = model(x)
    assert out.shape == (2, 1, 21, 21)


def test_forward_clamp_output_clips_to_range(srcnn_valid: SRCNN):
    # Drive Conv2d to produce something outside [0, 1] reliably.
    with torch.no_grad():
        for m in srcnn_valid.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.fill_(10.0)
                m.bias.fill_(5.0)
    x = torch.ones(1, 3, 33, 33)
    out = srcnn_valid(x, clamp_output=True, clamp_minmax=(0.0, 1.0))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_check_architecture_kernel_count_mismatch():
    # num_filters has 1 element but kernel_sizes has 3 — should be 2 vs 3.
    with pytest.raises(ValueError):
        SRCNN(num_channels=3, num_filters=(64,), kernel_sizes=(9, 1, 5))


def test_check_architecture_zero_filter_raises_for_long_tuple():
    """Zero filter is rejected when num_filters has >= 3 elements (the
    `any(f < 1)` branch fires)."""
    with pytest.raises(ValueError):
        SRCNN(num_channels=3, num_filters=(64, 32, 0), kernel_sizes=(9, 1, 5, 5))


def test_check_architecture_zero_filter_raises_for_short_tuple():
    """A zero filter is rejected at any length — the positive-value check runs
    independently of the length gate."""
    with pytest.raises(ValueError):
        SRCNN(num_channels=3, num_filters=(64, 0), kernel_sizes=(9, 1, 5))


def test_check_architecture_non_tuple_raises():
    with pytest.raises(ValueError):
        SRCNN(num_channels=3, num_filters=[64, 32], kernel_sizes=(9, 1, 5))


def test_check_architecture_empty_tuple_raises():
    with pytest.raises(ValueError):
        SRCNN(num_channels=3, num_filters=(), kernel_sizes=())


def test_hparams_property_round_trips():
    model = SRCNN(
        num_channels=3,
        num_filters=(64, 32),
        kernel_sizes=(9, 1, 5),
        padding=0,
    )
    h = model.hparams
    assert h["num_channels"] == 3
    assert h["num_filters"] == (64, 32)
    assert h["kernel_sizes"] == (9, 1, 5)
    assert h["padding"] == 0


def test_reset_parameters_zeroes_bias_and_sets_weight_std():
    """Verifies the SRCNN-paper init: weights ~ N(mean, std), biases = 0.
    After the init-args migration, reset_parameters is invoked explicitly."""
    torch.manual_seed(0)
    model = SRCNN(
        num_channels=3,
        num_filters=(64, 32),
        kernel_sizes=(9, 1, 5),
        padding=0,
    )
    model.reset_parameters(mean=0.0, std=0.01)
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            assert torch.allclose(m.bias, torch.zeros_like(m.bias))
            assert 0.005 < m.weight.std().item() < 0.02


def test_reset_parameters_default_std_matches_paper_not_tenfold_larger():
    """Dong et al. Sec. Training specifies std=0.001 for the weight-init
    Gaussian; a 10x-too-large default (0.01, previously the bug here) inflates
    variance 100x and would land the realized std far outside this band.
    Weights are pooled across every Conv2d (~20k elements) so the sample std
    is tight enough that only a real std defect, not sampling noise, can fail
    this."""
    torch.manual_seed(0)
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    model.reset_parameters()  # exercise reset_parameters' own default, not an explicit std
    weights = torch.cat(
        [m.weight.flatten() for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    )
    realized_std = weights.std().item()
    assert 0.0008 < realized_std < 0.0012, (
        f"expected ~0.001 (paper std); got {realized_std} -- the old 0.01 default "
        f"would land ~10x above this band"
    )


def test_init_only_args_rejected():
    """custom_init / init_mean / init_std were moved to SRCNNTrainingConfig."""
    with pytest.raises(TypeError):
        SRCNN(
            num_channels=3,
            num_filters=(64, 32),
            kernel_sizes=(9, 1, 5),
            padding=0,
            custom_init=True,
        )


def test_hparams_architectural_only():
    """After the init-args migration, SRCNN.hparams holds only architectural keys."""
    model = SRCNN(
        num_channels=3,
        num_filters=(64, 32),
        kernel_sizes=(9, 1, 5),
        padding=0,
    )
    assert set(model.hparams.keys()) == {
        "num_channels",
        "num_filters",
        "kernel_sizes",
        "padding",
        "eval_padding",
        "eval_padding_mode",
    }


def test_forward_multilayer_mapping_builds_extra_conv_and_preserves_shape():
    """A 3-entry num_filters (64, 32, 16) exercises the mapping comprehension
    for TWO hidden convs — every other valid config uses (64, 32) -> a single
    mapping conv, so the loop over range(len(num_filters) - 1) is untested for
    >1 layer. Asserts the deep net builds exactly 4 Conv2d (feat + 2 mapping +
    recon), that `mapping` holds 2 of them, and that same padding preserves the
    spatial size through the forward pass."""
    model = SRCNN(
        num_channels=3,
        num_filters=(64, 32, 16),
        kernel_sizes=(9, 1, 3, 5),  # len == len(num_filters) + 1
        padding="same",
    )
    n_convs = sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    assert n_convs == 4
    # (64,32) would give a single mapping conv; (64,32,16) must give two.
    assert sum(1 for m in model.mapping if isinstance(m, torch.nn.Conv2d)) == 2
    x = torch.zeros(2, 3, 16, 16)
    out = model(x)
    assert out.shape == (2, 3, 16, 16)


def test_variant_tag_is_the_kernel_triple():
    """The paper names its variants by kernel sizes, so the artifact does too."""
    assert SRCNN(num_channels=1, num_filters=(64, 32), kernel_sizes=(9, 1, 5)).variant_tag == "915"
    assert SRCNN(num_channels=1, num_filters=(64, 32), kernel_sizes=(9, 5, 5)).variant_tag == "955"


# ---------------------------------------------------------------------------
# eval_padding — the authors' test-time path (train valid, test same+replicate)
# ---------------------------------------------------------------------------


@pytest.fixture
def srcnn_authors() -> SRCNN:
    """The authors' released setup: `pad: 0` at train, same+replicate at test."""
    return SRCNN(
        num_channels=1,
        num_filters=(64, 32),
        kernel_sizes=(9, 1, 5),
        padding=0,
        eval_padding="same",
        eval_padding_mode="replicate",
    )


def test_eval_padding_only_applies_out_of_training_mode(srcnn_authors: SRCNN):
    """Train shrinks by 12 px as before; eval returns the full size."""
    x = torch.zeros(2, 1, 33, 33)

    srcnn_authors.train()
    assert srcnn_authors(x).shape == (2, 1, 21, 21)

    srcnn_authors.eval()
    assert srcnn_authors(x).shape == (2, 1, 33, 33)


def test_eval_padding_none_keeps_valid_in_both_modes(srcnn_valid: SRCNN):
    """The opt-out: `eval_padding=None` (the default) is valid convolution throughout."""
    x = torch.zeros(2, 3, 33, 33)
    for mode in (srcnn_valid.train, srcnn_valid.eval):
        mode()
        assert srcnn_valid(x).shape == (2, 3, 21, 21)


def test_eval_padding_pads_per_layer_not_the_input_once(srcnn_authors: SRCNN):
    """Replicate padding is re-derived per layer, as `SRCNN.m`'s imfilter is.

    Pre-padding the input by the full 6 px receptive field and running valid
    convolutions is the cheap-looking equivalent, and it is a different
    function: layer 2 would then see a border computed from replicated
    *input* pixels rather than replicated *layer-1 features*.
    """
    torch.manual_seed(0)
    srcnn_authors.reset_parameters(std=0.2)
    srcnn_authors.eval()
    x = torch.rand(1, 1, 24, 24)

    with torch.no_grad():
        per_layer = srcnn_authors(x)
        prepadded_input = torch.nn.functional.pad(x, (6, 6, 6, 6), mode="replicate")
        srcnn_authors.eval_padding = None
        prepadded = srcnn_authors(prepadded_input)

    assert per_layer.shape == prepadded.shape
    assert not torch.allclose(per_layer, prepadded, atol=1e-6)


def test_eval_padding_matches_an_explicit_per_layer_reference(srcnn_authors: SRCNN):
    """Pin the exact arithmetic against a hand-written pad-then-valid-conv loop."""
    torch.manual_seed(0)
    srcnn_authors.reset_parameters(std=0.2)
    srcnn_authors.eval()
    x = torch.rand(1, 1, 20, 20)

    layers = [*srcnn_authors.feat, *srcnn_authors.mapping, srcnn_authors.recon]
    with torch.no_grad():
        expected = x
        for layer in layers:
            if isinstance(layer, torch.nn.Conv2d):
                pad = (layer.kernel_size[0] - 1) // 2
                expected = torch.nn.functional.pad(expected, (pad, pad, pad, pad), mode="replicate")
                expected = torch.nn.functional.conv2d(expected, layer.weight, layer.bias)
            else:
                expected = layer(expected)
        actual = srcnn_authors(x)

    torch.testing.assert_close(actual, expected)


def test_eval_padding_is_recorded_in_hparams(srcnn_authors: SRCNN):
    """Checkpoints must carry it — it changes what the model outputs at eval."""
    assert srcnn_authors.hparams["eval_padding"] == "same"
    assert srcnn_authors.hparams["eval_padding_mode"] == "replicate"


def test_eval_padding_rejects_an_even_kernel():
    """`same` is not expressible for an even kernel without an asymmetric pad."""
    with pytest.raises(ValueError, match="odd kernel"):
        SRCNN(
            num_channels=1,
            num_filters=(64, 32),
            kernel_sizes=(9, 1, 4),
            padding=0,
            eval_padding="same",
        )
