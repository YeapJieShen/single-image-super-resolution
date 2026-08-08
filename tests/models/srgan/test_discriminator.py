import pytest
import torch

from sisr.models.srgan import SRDiscriminator


def test_output_is_one_logit_per_image():
    d = SRDiscriminator(hr_input_size=96)
    out = d(torch.randn(4, 3, 96, 96))
    assert out.shape == (4, 1)


def test_output_is_logits_not_probabilities():
    """No sigmoid: the head is paired with BCEWithLogitsLoss, which is the same
    objective and numerically stable. A restored sigmoid would double-activate."""
    d = SRDiscriminator(hr_input_size=96)
    out = d(torch.randn(64, 3, 96, 96) * 50)
    assert (out < 0).any() or (out > 1).any()


@pytest.mark.parametrize(("size", "expected"), [(96, 512 * 6 * 6), (128, 512 * 8 * 8)])
def test_dense_in_features_derived_from_declared_input_size(size, expected):
    """Four stride-2 convs => s = hr_input_size // 16."""
    d = SRDiscriminator(hr_input_size=size)
    assert d.classifier[0].in_features == expected


def test_dense_features_is_configurable():
    d = SRDiscriminator(hr_input_size=96, dense_features=512)
    assert d.classifier[0].out_features == 512


def test_indivisible_input_size_rejected():
    with pytest.raises(ValueError, match="divisible by 16"):
        SRDiscriminator(hr_input_size=100)


def test_hparams_roundtrip_for_metadata():
    d = SRDiscriminator(hr_input_size=96, dense_features=512)
    assert d.hparams["hr_input_size"] == 96
    assert d.hparams["dense_features"] == 512
