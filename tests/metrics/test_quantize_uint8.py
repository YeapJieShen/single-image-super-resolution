"""quantize_uint8: PSNR rounds to 8-bit before scoring, matching Dong et al.'s
demo_SR.m (issue #276). Scoped to PSNR only -- Dong's script never quantizes
for SSIM, and this project's own eval convention already handles SSIM.
"""

import pytest
import torch

from sisr.metrics.scoring import SRScorer
from sisr.models.srcnn import SRCNNEvalConfig
from sisr.training.config import SREvalConfig

# General PSNR fixture: neither value sits on a uint8 grid line.
# hr*255 = [128.01, 76.755, 178.245], sr*255 = [129.7695, 74.5875, 180.4125],
# so quantizing changes both. Quantized levels (ties away from zero):
# hr -> [128, 77, 178], sr -> [130, 75, 180].
# Values computed via torchmetrics.functional.image.peak_signal_noise_ratio.
HR = torch.tensor([[[[0.502, 0.301, 0.699]]]])
SR = torch.tensor([[[[0.5089, 0.2925, 0.7075]]]])
FLOAT_PSNR = 41.93571091
QUANTIZED_PSNR = 42.11020279


def _psnr_scorer(quantize_uint8: bool) -> SRScorer:
    return SRScorer(
        SREvalConfig(
            crop_border=0,
            psnr_channels=["RGB"],
            ssim_channels=[],
            quantize_uint8=quantize_uint8,
        )
    )


def _ssim_scorer(quantize_uint8: bool) -> SRScorer:
    return SRScorer(
        SREvalConfig(
            crop_border=0,
            psnr_channels=[],
            ssim_channels=["RGB"],
            quantize_uint8=quantize_uint8,
        )
    )


# SSIM's default gaussian window needs a real-sized image (11x11+); HR/SR above
# are 1x3 and only meant for the PSNR literal, so the SSIM test gets its own,
# larger, fixed-seed pair. (A 1x3 image raises a torch padding RuntimeError
# under SSIM -- this is why the PSNR and SSIM checks use separate scorers/
# fixtures rather than one shared "RGB in both channels" config.)
_g = torch.Generator().manual_seed(276)
SSIM_HR = torch.rand(1, 3, 32, 32, generator=_g)
SSIM_SR = (SSIM_HR + 0.05 * (torch.rand(1, 3, 32, 32, generator=_g) - 0.5)).clamp(0, 1)


def test_quantize_uint8_defaults_off_on_base_config():
    assert SREvalConfig().quantize_uint8 is False


def test_quantize_uint8_defaults_on_for_srcnn():
    assert SRCNNEvalConfig().quantize_uint8 is True


def test_quantize_uint8_true_rounds_before_psnr():
    """RED on pre-fix main: SREvalConfig has no quantize_uint8 field at all
    (TypeError constructing SREvalConfig(quantize_uint8=...)).

    Mutation this guards: quantize_uint8=True accepted but never applied (a
    no-op flag) -- scored_on would then equal FLOAT_PSNR instead of
    QUANTIZED_PSNR, and the final != assertion would also stop failing.
    """
    scored_off = _psnr_scorer(False).score(SR, HR).psnr["RGB"].item()
    scored_on = _psnr_scorer(True).score(SR, HR).psnr["RGB"].item()

    assert scored_off == pytest.approx(FLOAT_PSNR, abs=1e-4)
    assert scored_on == pytest.approx(QUANTIZED_PSNR, abs=1e-4)
    assert scored_on != scored_off


def test_quantize_uint8_does_not_affect_ssim():
    """Mutation this guards: quantizing inside metric_tensors's shared dict
    (rather than only the tensors handed to psnr()) -- SSIM would then also
    change between the two configs below, and this equality would fail.
    """
    ssim_off = _ssim_scorer(False).score(SSIM_SR, SSIM_HR).ssim["RGB"].item()
    ssim_on = _ssim_scorer(True).score(SSIM_SR, SSIM_HR).ssim["RGB"].item()
    assert ssim_on == ssim_off


def test_quantize_uint8_ties_away_from_zero():
    """Mutations M3 (ties-to-even) and M8 (ties half-down) both fail this.

    A tie occurs when x*255 = k + 0.5 for integer k. This fixture has one tie
    (hr) and one non-tie (sr) so both M3 and M8 diverge from the correct
    ties-away-from-zero convention in the same way.

    Fixture (in float64 for exact ties/non-ties):
      hr = 126.5/255  ⟹  hr*255 = 126.5 (tie)
      sr = 129.0/255  ⟹  sr*255 = 129.0 (non-tie, integer)

    Correct rounding (ties away from zero):
      hr*255 = 126.5  ⟹ rounds away from zero to 127
      sr*255 = 129.0  ⟹ rounds to 129
      MSE = ((129 - 127)/255)² = (2/255)²
      ⟹ PSNR = 20*log10(255/2) = 42.110204 dB

    M3 (ties-to-even: torch.round):
      hr*255 = 126.5  ⟹ rounds to 126 (nearest even)
      sr*255 = 129.0  ⟹ rounds to 129
      MSE = ((129 - 126)/255)² = (3/255)²
      ⟹ PSNR = 20*log10(255/3) = 38.588379 dB (wrong)

    M8 (ties half-down: torch.ceil(x*255 - 0.5)):
      hr*255 = 126.5  ⟹ rounds down to 126
      sr*255 = 129.0  ⟹ rounds to 129
      MSE = ((129 - 126)/255)² = (3/255)²
      ⟹ PSNR = 20*log10(255/3) = 38.588379 dB (same wrong result as M3)
    """
    # Fixture in float64 so 126.5/255 and 129.0/255 are exact
    hr_tie = torch.tensor([[[[126.5 / 255.0]]]], dtype=torch.float64)
    sr_nontie = torch.tensor([[[[129.0 / 255.0]]]], dtype=torch.float64)

    # Correct rounding (away from zero): 127 - 129 -> MSE=(2/255)² -> PSNR=42.11 dB
    expected_psnr_correct = 42.110204

    scored_on = _psnr_scorer(True).score(sr_nontie, hr_tie).psnr["RGB"].item()

    # With quantization via ties-away-from-zero, we expect 42.11 dB.
    # Both M3 (ties-to-even) and M8 (ties half-down) would give 38.59 dB,
    # so either mutation fails this assertion.
    assert scored_on == pytest.approx(expected_psnr_correct, abs=1e-4)
