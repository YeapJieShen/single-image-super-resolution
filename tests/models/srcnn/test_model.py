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
    assert set(model.hparams.keys()) == {"num_channels", "num_filters", "kernel_sizes", "padding"}
