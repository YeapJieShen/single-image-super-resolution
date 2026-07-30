import torch

from sisr.colorspace import rgb_to_ycbcr, ycbcr_to_rgb


def test_rgb_to_ycbcr_shape_and_dtype():
    x = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    y = rgb_to_ycbcr(x)
    assert y.shape == (2, 3, 8, 8)
    assert y.dtype == torch.float32


def test_ycbcr_to_rgb_shape_and_dtype():
    x = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    y = ycbcr_to_rgb(x)
    assert y.shape == (2, 3, 8, 8)
    assert y.dtype == torch.float32


def test_rgb_to_ycbcr_known_values_white():
    # White: (1, 1, 1) -> Y=1, Cb=0.5, Cr=0.5
    x = torch.ones(1, 3, 1, 1)
    y = rgb_to_ycbcr(x)
    expected = torch.tensor([[[[1.0]], [[0.5]], [[0.5]]]])
    assert torch.allclose(y, expected, atol=1e-5)


def test_rgb_to_ycbcr_known_values_black():
    # Black: (0, 0, 0) -> Y=0, Cb=0.5, Cr=0.5
    x = torch.zeros(1, 3, 1, 1)
    y = rgb_to_ycbcr(x)
    expected = torch.tensor([[[[0.0]], [[0.5]], [[0.5]]]])
    assert torch.allclose(y, expected, atol=1e-5)


def test_rgb_to_ycbcr_known_values_red():
    # Pure red: (1, 0, 0) -> Y=0.299, Cb=0.5−0.169, Cr=0.5+0.500
    x = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
    y = rgb_to_ycbcr(x)
    expected = torch.tensor([[[[0.299]], [[0.331]], [[1.000]]]])
    assert torch.allclose(y, expected, atol=1e-5)


def test_round_trip_within_coefficient_precision():
    # BT.601 coefficients are 3-decimal-place; round-trip error floor ~5e-4.
    torch.manual_seed(0)
    x = torch.rand(1, 3, 16, 16)
    y = ycbcr_to_rgb(rgb_to_ycbcr(x))
    err = (y - x).abs().max().item()
    assert err < 5e-4
