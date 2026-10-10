"""The scoring-side switches (8-bit rounding, grey-plane scoring) and what each template sets."""

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
EXPECTED = {  # name -> (round_to_uint8, grey_plane_scoring, float_degrade on every eval dataset)
    "vdsr": (True, True, True),
    "drcn": (True, True, True),
    "edsr": (True, False, False),
    "mdsr": (True, True, False),
    "srcnn": (True, False, False),
    "srresnet": (False, False, False),
    "srgan": (False, False, False),
    "espcn": (False, False, False),
}


def _eval_datasets(cfg):
    data = cfg["data"]
    yield data["val_dataset"]
    yield from data.get("test_datasets", {}).values()


@pytest.mark.parametrize("name", list(EXPECTED))
def test_each_template_sets_its_papers_evaluation_protocol(name):
    cfg = yaml.safe_load((TEMPLATES / f"config.{name}.template.yaml").read_text(encoding="ascii"))
    rnd, grey, flt = EXPECTED[name]
    ev = cfg["model"]["init_args"].get("eval_config", {}).get("init_args", {})

    assert ev.get("round_to_uint8", False) is rnd
    assert ev.get("grey_plane_scoring", False) is grey
    for ds in _eval_datasets(cfg):
        assert ds.get("init_args", {}).get("float_degrade", False) is flt
        assert ds.get("init_args", {}).get("yield_is_grey", False) is grey  # flag iff switch


# --- grey-plane scoring: greyness is the file mode the dataset reports, never R == G == B ------

STUDIO_GAIN = 20 * math.log10(255 / 219)  # studio-range Y compresses errors by 219/255


def _grey(seed, n=1, size=16):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, size, size, generator=g).expand(n, 3, size, size).clone()


def _flag(*values):
    return torch.tensor(values)


def _gy(scorer, sr, hr, grey):
    return float(scorer.score(sr, hr, is_grey=grey).psnr["Y"])


def _grey_scorer(on=True):
    return _scorer(grey_plane_scoring=on)


def test_a_grey_file_scores_on_its_raw_plane_which_is_the_studio_gain_lower():
    hr = _grey(0)
    sr = (hr + 0.02 * _grey(1)).clamp(0, 1)
    off = _gy(_grey_scorer(False), sr, hr, None)
    on = _gy(_grey_scorer(), sr, hr, _flag(True))

    assert off - on == pytest.approx(STUDIO_GAIN, abs=2e-3)


def test_an_rgb_file_with_equal_channels_is_colour_like_set14_man():
    hr = _grey(2)  # R == G == B everywhere, but the dataset says the file is not single-channel
    sr = (hr + 0.02).clamp(0, 1)

    assert _gy(_grey_scorer(), sr, hr, _flag(False)) == _gy(_grey_scorer(False), sr, hr, None)


def test_a_mixed_batch_is_decided_per_image_by_the_flag():
    hr = torch.cat([_grey(3), torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(4))])
    sr = (hr + torch.tensor([0.01, 0.05]).view(2, 1, 1, 1)).clamp(0, 1)  # unequal errors
    on = _grey_scorer()

    expected = (_gy(on, sr[:1], hr[:1], _flag(True)) + _gy(on, sr[1:], hr[1:], _flag(False))) / 2
    assert _gy(on, sr, hr, _flag(True, False)) == pytest.approx(expected, abs=1e-4)
    for wrong in ((True, True), (False, False)):  # one flagged image is half the gain
        assert _gy(on, sr, hr, _flag(*wrong)) != pytest.approx(expected, abs=0.1)


def test_the_switch_on_without_a_flag_is_refused_rather_than_guessed():
    hr = _grey(5)
    with pytest.raises(ValueError, match="is_grey"):
        _grey_scorer().score(hr, hr)


def test_a_flag_without_the_switch_changes_nothing():
    hr = _grey(6)
    sr = (hr + 0.02).clamp(0, 1)
    assert _gy(_grey_scorer(False), sr, hr, _flag(True)) == _gy(_grey_scorer(False), sr, hr, None)


def test_cb_cr_keys_are_not_touched_by_the_grey_flag():
    cfg = dict(crop_border=0, psnr_channels=["Cb", "Cr"], ssim_channels=[])
    hr = _grey(9)
    sr = (hr + 0.02).clamp(0, 1)
    a = SRScorer(SREvalConfig(grey_plane_scoring=True, **cfg)).score(sr, hr, is_grey=_flag(True))
    b = SRScorer(SREvalConfig(**cfg)).score(sr, hr)
    assert {k: float(v) for k, v in a.psnr.items()} == {k: float(v) for k, v in b.psnr.items()}


def test_validation_step_passes_the_dataset_flag_through_to_scoring():
    import functools

    from sisr.models.srresnet.model import SRResNet
    from sisr.processors import RGBProcessor
    from sisr.training import SRLightning, SRTrainingConfig

    hr = _grey(7)
    sr = (hr + 0.02).clamp(0, 1)

    def run(batch):
        m = SRLightning(
            model=SRResNet(scale=2, num_residual_blocks=1),
            processor=RGBProcessor(),
            training_config=SRTrainingConfig(scale=2),
            eval_config=SREvalConfig(
                crop_border=0, psnr_channels=["Y"], ssim_channels=[], grey_plane_scoring=True
            ),
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )
        logged = {}
        m.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
        m._step = lambda b: (torch.tensor(0.0), None, None, sr.clone(), hr.clone())
        m.validation_step(batch, 0)
        return logged["psnr/val/Y"]

    lr = torch.rand(1, 3, 8, 8)
    assert run((lr, hr, _flag(False))) - run((lr, hr, _flag(True))) == pytest.approx(
        STUDIO_GAIN, abs=2e-3
    )
