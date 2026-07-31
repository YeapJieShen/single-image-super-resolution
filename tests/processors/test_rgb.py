"""Behavior tests for RGBProcessor — no-op pass-through."""

import torch

from sisr.processors import RGBProcessor, SRProcessor


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
