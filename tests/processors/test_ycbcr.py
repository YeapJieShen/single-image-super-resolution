"""Behavior tests for YCbCrProcessor — full YCbCr convert."""
import torch

from sisr.processors import YCbCrProcessor, SRProcessor
from sisr.utils import rgb_to_ycbcr


def test_ycbcr_processor_is_srprocessor():
    assert isinstance(YCbCrProcessor(), SRProcessor)


def test_ycbcr_extract_returns_full_ycbcr():
    """extract returns full YCbCr (shape [B,3,H,W]) matching rgb_to_ycbcr."""
    p = YCbCrProcessor()
    lr = torch.rand(2, 3, 8, 8)
    out = p.extract(lr)
    assert out.shape == (2, 3, 8, 8)
    assert torch.allclose(out, rgb_to_ycbcr(lr), atol=1e-6)


def test_ycbcr_reconstruct_converts_back_to_rgb():
    """reconstruct converts YCbCr SR output to RGB."""
    p = YCbCrProcessor()
    lr = torch.rand(2, 3, 8, 8)
    sr_ycbcr = rgb_to_ycbcr(lr)        # use a known-valid YCbCr tensor
    out = p.reconstruct(sr_ycbcr, lr)
    assert out.shape == (2, 3, 8, 8)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_ycbcr_roundtrip_recovers_input():
    """extract -> reconstruct approximately recovers the input RGB.

    Tolerance is 1e-3 because the BT.601 coefficients in :mod:`sisr.utils`
    are truncated to 3 decimal places (e.g. ``-0.169`` vs the exact
    ``-0.168736``), so the forward and inverse matrices aren't perfect
    inverses. Observed worst-case error is ~5e-4 across the chroma channels.
    """
    p = YCbCrProcessor()
    lr = torch.rand(2, 3, 8, 8)
    roundtrip = p.reconstruct(p.extract(lr), lr_rgb=lr)
    assert torch.allclose(roundtrip, lr, atol=1e-3)
