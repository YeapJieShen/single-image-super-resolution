"""The scoring-side 8-bit rounding switch, and what each template sets."""

import math
from pathlib import Path

import pytest
import torch
import yaml

from sisr.metrics.scoring import SRScorer
from sisr.training import SREvalConfig


def _scorer(**kw):
    return SRScorer(SREvalConfig(crop_border=0, psnr_channels=["Y"], ssim_channels=[], **kw))


def test_rounding_scores_the_8_bit_plane_and_off_scores_the_raw_float():
    hr = torch.zeros(1, 3, 8, 8)
    cfg = dict(crop_border=0, psnr_channels=["RGB"], ssim_channels=[])
    sr_low, sr_high = torch.full_like(hr, 0.4 / 255), torch.full_like(hr, 0.6 / 255)
    on = SRScorer(SREvalConfig(round_to_uint8=True, **cfg))
    off = SRScorer(SREvalConfig(**cfg))

    assert float(off.score(sr_low, hr).psnr["RGB"]) == pytest.approx(
        20 * math.log10(255 / 0.4), abs=1e-3
    )
    assert math.isinf(float(on.score(sr_low, hr).psnr["RGB"]))  # 0.4 level rounds to 0
    assert float(on.score(sr_high, hr).psnr["RGB"]) == pytest.approx(
        20 * math.log10(255), abs=1e-3
    )  # 0.6 -> 1


# --- what each paper's template sets ------------------------------------------------------

TEMPLATES = Path(__file__).resolve().parents[2] / "templates"
EXPECTED = {  # name -> (round_to_uint8, float_degrade on every eval dataset)
    "vdsr": (True, True),
    "drcn": (True, True),
    "edsr": (True, False),
    "mdsr": (True, False),
    "srcnn": (True, False),
    "srresnet": (False, False),
    "srgan": (False, False),
    "espcn": (False, False),
}


def _eval_datasets(cfg):
    data = cfg["data"]
    yield data["val_dataset"]
    yield from data.get("test_datasets", {}).values()


@pytest.mark.parametrize("name", list(EXPECTED))
def test_each_template_sets_its_papers_evaluation_protocol(name):
    cfg = yaml.safe_load((TEMPLATES / f"config.{name}.template.yaml").read_text(encoding="ascii"))
    rnd, flt = EXPECTED[name]
    ev = cfg["model"]["init_args"].get("eval_config", {}).get("init_args", {})

    assert ev.get("round_to_uint8", False) is rnd
    for ds in _eval_datasets(cfg):
        assert ds.get("init_args", {}).get("float_degrade", False) is flt
