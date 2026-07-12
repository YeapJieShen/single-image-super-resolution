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


def test_ycbcr_reconstruct_matches_bt601_inverse_not_just_clamp():
    """reconstruct must apply the real BT.601 full-range inverse, not merely
    land in [0, 1]. The old assertion (out.min() >= 0, out.max() <= 1) passed
    even for wrong coefficients because ycbcr_to_rgb clamps to [0, 1]. Here the
    chroma is mid-range so the correct RGB is strictly inside (0, 1) — the clamp
    is a genuine no-op, and a coefficient/sign/offset regression changes the
    values instead of being hidden."""
    p = YCbCrProcessor()
    # Stored Y=0.5, Cb=0.6, Cr=0.4 -> signed cb=+0.1, cr=-0.1 (the +0.5 offset).
    ycbcr = torch.tensor([0.5, 0.6, 0.4]).view(1, 3, 1, 1).expand(1, 3, 2, 2).clone()
    lr = torch.rand(1, 3, 2, 2)  # only its shape matters to reconstruct
    out = p.reconstruct(ycbcr, lr)

    # Hand-computed BT.601 full-range inverse (coefficients from sisr.utils).
    cb, cr = 0.1, -0.1
    r = 0.5 + 1.402 * cr
    g = 0.5 - 0.344 * cb - 0.714 * cr
    b = 0.5 + 1.772 * cb
    expected = torch.tensor([r, g, b]).view(1, 3, 1, 1).expand(1, 3, 2, 2)

    assert out.min() > 0.0 and out.max() < 1.0  # clamp is a genuine no-op here
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


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
