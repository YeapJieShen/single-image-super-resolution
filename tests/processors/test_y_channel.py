"""Behavior tests for YChannelProcessor — Y extract + bicubic Cb/Cr stitch."""
import torch

from sisr.processors import YChannelProcessor, SRProcessor
from sisr.utils import rgb_to_ycbcr, ycbcr_to_rgb


def test_y_channel_processor_is_srprocessor():
    assert isinstance(YChannelProcessor(), SRProcessor)


def test_y_channel_extract_returns_single_y_channel():
    """extract returns Y of the LR in YCbCr (shape [B,1,H,W])."""
    p = YChannelProcessor()
    lr = torch.rand(2, 3, 8, 8)
    out = p.extract(lr)
    assert out.shape == (2, 1, 8, 8)
    expected = rgb_to_ycbcr(lr)[:, 0:1]
    assert torch.allclose(out, expected, atol=1e-6)


def test_y_channel_reconstruct_same_size_roundtrips_to_lr():
    """SRCNN-style (SR-Y same size as LR): feeding the LR's own Y back through
    reconstruct must roundtrip to the original RGB. Same-size bicubic Cb/Cr
    resampling is exact identity, so out == ycbcr_to_rgb(rgb_to_ycbcr(lr)) ~= lr.
    The old assertion only checked out was in [0, 1] (always true after the
    clamp); this pins that the stitch + inverse actually recover the input."""
    import torch.nn.functional as F
    p = YChannelProcessor()
    lr = torch.rand(2, 3, 16, 16)
    sr_y = rgb_to_ycbcr(lr)[:, 0:1]        # SR-Y is exactly the LR-Y
    out = p.reconstruct(sr_y, lr)
    assert out.shape == (2, 3, 16, 16)
    # Same-size interpolate is an exact no-op, so the whole path is a roundtrip.
    same = F.interpolate(rgb_to_ycbcr(lr)[:, 1:], size=(16, 16),
                         mode="bicubic", align_corners=False)
    assert torch.equal(same, rgb_to_ycbcr(lr)[:, 1:])  # bicubic@same-size == identity
    # Tolerance 1e-3: the BT.601 coeffs in sisr.utils are truncated to 3 dp, so
    # the forward/inverse matrices aren't perfect inverses (worst-case ~5e-4).
    assert torch.allclose(out, lr, atol=1e-3)


def test_y_channel_reconstruct_upscaled_stitches_bicubic_chroma():
    """SRResNet-style (SR-Y larger than LR): the Y passes through untouched and
    the LR Cb/Cr are bicubic-upsampled to the SR-Y size, then converted to RGB.
    The old test asserted shape only. This re-derives the exact reconstruction
    from the same primitives, so switching the interpolation mode/target size or
    dropping the Y pass-through fails."""
    import torch.nn.functional as F
    p = YChannelProcessor()
    lr = torch.rand(2, 3, 8, 8)
    sr_y = torch.rand(2, 1, 32, 32)        # 4x upscale
    out = p.reconstruct(sr_y, lr)
    assert out.shape == (2, 3, 32, 32)

    lr_ycbcr = rgb_to_ycbcr(lr)
    cbcr = F.interpolate(lr_ycbcr[:, 1:], size=(32, 32),
                        mode="bicubic", align_corners=False)
    expected = ycbcr_to_rgb(torch.cat([sr_y, cbcr], dim=1))
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)

