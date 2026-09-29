import pytest
import torch

from sisr.models.espcn import ESPCN


def test_package_reexports_public_symbols():
    from sisr.models.espcn import ESPCN as PkgESPCN
    from sisr.models.espcn import ESPCNEvalConfig, ESPCNTrainingConfig
    from sisr.models.espcn.config import ESPCNEvalConfig as E
    from sisr.models.espcn.config import ESPCNTrainingConfig as T
    from sisr.models.espcn.model import ESPCN as M

    assert PkgESPCN is M and ESPCNTrainingConfig is T and ESPCNEvalConfig is E


def test_input_contract_is_native_lr():
    assert ESPCN.input_contract == "native_lr"


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("scale", [1, 2, 3, 4])
def test_output_is_scale_times_input(scale, channels):
    model = ESPCN(scale=scale, in_out_channels=channels)
    out = model(torch.zeros(2, channels, 9, 11))
    assert out.shape == (2, channels, 9 * scale, 11 * scale)


def test_exactly_three_convs_tanh_hidden_and_no_trailing_activation():
    model = ESPCN(scale=3)
    layers = list(model.net)
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 3
    assert [type(m) for m in layers] == [
        torch.nn.Conv2d,
        torch.nn.Tanh,
        torch.nn.Conv2d,
        torch.nn.Tanh,
        torch.nn.Conv2d,
        torch.nn.PixelShuffle,
    ]
    assert [(c.in_channels, c.out_channels, c.kernel_size) for c in convs] == [
        (1, 64, (5, 5)),
        (64, 32, (3, 3)),
        (32, 9, (3, 3)),
    ]


def test_hparams_round_trip_rebuilds_identical_shapes():
    model = ESPCN(scale=2, in_out_channels=3, filters=[16, 8], kernel_sizes=[5, 3, 3])
    rebuilt = ESPCN(**model.hparams)
    assert {k: v.shape for k, v in model.state_dict().items()} == {
        k: v.shape for k, v in rebuilt.state_dict().items()
    }
    assert model.hparams["filters"] == (16, 8)
    assert model.hparams["kernel_sizes"] == (5, 3, 3)


def test_variant_tag():
    assert ESPCN(scale=3).variant_tag == "5-3-3"
    assert ESPCN(scale=3, filters=(8, 8), kernel_sizes=(3, 3, 3)).variant_tag == "3-3-3"


@pytest.mark.parametrize("scale", [0, -1, 2.5, True])
def test_bad_scale_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        ESPCN(scale=scale)


def test_kernel_sizes_length_must_match_filters():
    with pytest.raises(ValueError, match="kernel_sizes"):
        ESPCN(scale=2, kernel_sizes=(5, 3))


def test_non_positive_kernel_rejected():
    with pytest.raises(ValueError, match="kernel_sizes"):
        ESPCN(scale=2, kernel_sizes=(5, 0, 3))


def test_even_kernels_rejected_because_same_padding_would_misalign():
    with pytest.raises(ValueError, match="odd"):
        ESPCN(scale=2, kernel_sizes=(4, 3, 3))
