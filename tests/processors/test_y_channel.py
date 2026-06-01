"""Behavior tests for YChannelProcessor — Y extract + bicubic Cb/Cr stitch."""
import torch

from sisr.processors import YChannelProcessor, SRProcessor
from sisr.utils import rgb_to_ycbcr


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


def test_y_channel_reconstruct_same_size_lr_and_sr():
    """SR-Y same spatial size as LR (SRCNN-style): output shape [B,3,H,W], valid RGB."""
    p = YChannelProcessor()
    lr = torch.rand(2, 3, 16, 16)
    sr_y = rgb_to_ycbcr(lr)[:, 0:1]    # SR-Y exactly the LR-Y (round-trip)
    out = p.reconstruct(sr_y, lr)
    assert out.shape == (2, 3, 16, 16)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_y_channel_reconstruct_upscaled_sr():
    """SR-Y larger than LR (SRResNet-style): bicubic Cb/Cr is interpolated up to SR-Y size."""
    p = YChannelProcessor()
    lr = torch.rand(2, 3, 8, 8)
    sr_y = torch.rand(2, 1, 32, 32)    # 4x upscale
    out = p.reconstruct(sr_y, lr)
    assert out.shape == (2, 3, 32, 32)


def test_y_channel_reconstruct_y_channel_preserved():
    """Reconstructed RGB's Y channel matches the input sr_y after YCbCr roundtrip."""
    p = YChannelProcessor()
    lr = torch.rand(2, 3, 16, 16)
    sr_y = torch.rand(2, 1, 16, 16)
    out_rgb = p.reconstruct(sr_y, lr)
    recovered_y = rgb_to_ycbcr(out_rgb)[:, 0:1]
    assert torch.allclose(recovered_y, sr_y, atol=1e-4)
