import math

import pytest
import torch

from sisr.colorspace import rgb_to_ycbcr, rgb_to_ycbcr_studio, ycbcr_to_rgb


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


# ---------------------------------------------------------------------------
# rgb_to_ycbcr_studio (P2.8) — BT.601 studio range, metric-only
# ---------------------------------------------------------------------------


def test_rgb_to_ycbcr_studio_shape_and_dtype():
    x = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    y = rgb_to_ycbcr_studio(x)
    assert y.shape == (2, 3, 8, 8)
    assert y.dtype == torch.float32


def test_rgb_to_ycbcr_studio_known_values_white():
    # White: full-range Y=1, Cb=Cr=0.5 -> studio Y=16/255+219/255=235/255,
    # Cb=Cr=128/255 (chroma stays centered — white has no color).
    x = torch.ones(1, 3, 1, 1)
    y = rgb_to_ycbcr_studio(x)
    expected = torch.tensor([[[[235 / 255]], [[128 / 255]], [[128 / 255]]]])
    assert torch.allclose(y, expected, atol=1e-6)


def test_rgb_to_ycbcr_studio_known_values_black():
    # Black: full-range Y=0 -> studio Y=16/255 (the legal-range floor).
    x = torch.zeros(1, 3, 1, 1)
    y = rgb_to_ycbcr_studio(x)
    expected = torch.tensor([[[[16 / 255]], [[128 / 255]], [[128 / 255]]]])
    assert torch.allclose(y, expected, atol=1e-6)


def test_rgb_to_ycbcr_studio_differs_from_full_range():
    """Hard constraint check: the studio-range conversion is a distinct
    function, not an in-place change to rgb_to_ycbcr's behaviour — see
    test_rgb_to_ycbcr_known_values_white/black for proof rgb_to_ycbcr
    itself is unchanged."""
    x = torch.rand(1, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(rgb_to_ycbcr(x), rgb_to_ycbcr_studio(x))


def _psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.mean((a - b) ** 2)
    return 10 * torch.log10(torch.tensor(data_range**2) / mse)


def test_y_channel_studio_psnr_offset_matches_algebraic_identity():
    """The decisive test (P2.8): studio-range Y is an affine rescale of
    full-range Y by a constant factor (219/255, offset 16/255) — the diff
    (and hence MSE) scales by (219/255)**2 regardless of the actual image
    content, so PSNR_studio - PSNR_full is the *exact* constant
    20*log10(255/219) for any pair of images with nonzero error. This is the
    algebraic fact the empirical Set5/Set14 corroboration in triage P2.8
    (30.56 raw + 1.3219 ~= 31.88 vs. the paper's 32.05) depends on.
    """
    expected_delta = 20 * math.log10(255 / 219)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        sr = torch.rand(2, 3, 16, 16, generator=g)
        hr = torch.rand(2, 3, 16, 16, generator=g)

        sr_y_full = rgb_to_ycbcr(sr)[:, 0:1]
        hr_y_full = rgb_to_ycbcr(hr)[:, 0:1]
        sr_y_studio = rgb_to_ycbcr_studio(sr)[:, 0:1]
        hr_y_studio = rgb_to_ycbcr_studio(hr)[:, 0:1]

        psnr_full = _psnr(sr_y_full, hr_y_full)
        psnr_studio = _psnr(sr_y_studio, hr_y_studio)

        delta = (psnr_studio - psnr_full).item()
        assert delta == pytest.approx(expected_delta, abs=1e-4)


def test_chroma_channels_use_the_224_scale_not_the_luma_scale():
    """BT.601's studio-range chroma legal range is [16, 240] (224/255), not
    [16, 235] (219/255) like luma — MATLAB's rgb2ycbcr and BasicSR's
    bgr2ycbcr both preserve this luma/chroma asymmetry. Cb/Cr's studio-full
    PSNR delta is therefore 20*log10(255/224), a *different* constant from
    Y's 20*log10(255/219) — collapsing the two to one constant would itself
    be a fidelity bug, so this locks the distinction in rather than assuming
    it."""
    expected_chroma_delta = 20 * math.log10(255 / 224)
    expected_luma_delta = 20 * math.log10(255 / 219)
    assert expected_chroma_delta != pytest.approx(expected_luma_delta, abs=1e-3)

    g = torch.Generator().manual_seed(1)
    sr = torch.rand(2, 3, 16, 16, generator=g)
    hr = torch.rand(2, 3, 16, 16, generator=g)

    for channel in (1, 2):  # Cb, Cr
        sr_full = rgb_to_ycbcr(sr)[:, channel : channel + 1]
        hr_full = rgb_to_ycbcr(hr)[:, channel : channel + 1]
        sr_studio = rgb_to_ycbcr_studio(sr)[:, channel : channel + 1]
        hr_studio = rgb_to_ycbcr_studio(hr)[:, channel : channel + 1]

        delta = (_psnr(sr_studio, hr_studio) - _psnr(sr_full, hr_full)).item()
        assert delta == pytest.approx(expected_chroma_delta, abs=1e-4)
