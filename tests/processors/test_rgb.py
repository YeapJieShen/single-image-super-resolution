"""Behavior tests for the RGB processors — pass-through and signed-output."""

import torch

from sisr.processors import RGBProcessor, RGBSignedOutputProcessor, SRProcessor


def test_rgb_processor_is_srprocessor():
    """RGBProcessor is a concrete SRProcessor subclass."""
    assert isinstance(RGBProcessor(), SRProcessor)


def test_rgb_extract_is_identity():
    """extract returns the input tensor unchanged (same object)."""
    p = RGBProcessor()
    lr = torch.rand(2, 3, 8, 8)
    out = p.extract(lr)
    assert out is lr


def test_rgb_reconstruct_is_identity():
    """reconstruct returns the model output unchanged; lr_rgb is ignored."""
    p = RGBProcessor()
    sr_rgb = torch.rand(2, 3, 16, 16)
    lr_rgb = torch.rand(2, 3, 8, 8)
    out = p.reconstruct(sr_rgb, lr_rgb)
    assert out is sr_rgb


def test_rgb_model_channels_is_3():
    assert RGBProcessor().model_channels == 3


def test_rgb_output_range_is_unit():
    assert RGBProcessor().output_range == (0.0, 1.0)


def test_rgb_extract_target_defaults_to_extract():
    """RGBProcessor doesn't override extract_target, so LR and HR share one transform."""
    p = RGBProcessor()
    hr = torch.rand(2, 3, 16, 16)
    assert p.extract_target(hr) is hr


def test_signed_output_extract_leaves_lr_in_unit_range():
    """The paper feeds the model LR in [0, 1] — extract must not rescale the input."""
    p = RGBSignedOutputProcessor()
    lr = torch.rand(2, 3, 8, 8)
    assert p.extract(lr) is lr


def test_signed_output_extract_target_maps_unit_to_signed():
    """extract_target sends [0, 1] onto [-1, 1] — the paper's HR range."""
    p = RGBSignedOutputProcessor()
    hr = torch.tensor([[[[0.0, 0.5, 1.0]]]]).expand(1, 3, 1, 3)
    expected = torch.tensor([[[[-1.0, 0.0, 1.0]]]]).expand(1, 3, 1, 3)
    assert torch.allclose(p.extract_target(hr), expected)


def test_signed_output_reconstruct_inverts_extract_target():
    """reconstruct must be the exact inverse of extract_target, or loss and metrics disagree."""
    p = RGBSignedOutputProcessor()
    hr = torch.rand(2, 3, 16, 16)
    roundtrip = p.reconstruct(p.extract_target(hr), lr_rgb=torch.rand(2, 3, 4, 4))
    assert torch.allclose(roundtrip, hr, atol=1e-7)


def test_signed_output_model_channels_is_3():
    assert RGBSignedOutputProcessor().model_channels == 3


def test_signed_output_is_srprocessor():
    assert isinstance(RGBSignedOutputProcessor(), SRProcessor)


def test_signed_output_output_range_is_signed():
    """The whole reason this processor exists: output_range is [-1, 1], not [0, 1]."""
    assert RGBSignedOutputProcessor().output_range == (-1.0, 1.0)
