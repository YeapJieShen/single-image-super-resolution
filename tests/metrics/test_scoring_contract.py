"""Golden-value contract for the validation scoring path.

Nothing else pins what ``validation_step`` *emits* — the existing metric tests
each check one property (a reduction, a routing, a key set) but none fixes the
tag grammar and the numbers together. That makes the scoring path safe to
"tidy" and silently renumber, which is exactly the hazard trap 2 describes:
a figure is comparable only to one computed the same way, and nothing here
would have noticed the difference.

Two deliberate design choices:

* **The model is factored out.** ``_step`` is replaced with fixed tensors, so
  every number below is a function of the metric math alone — not of torch's
  RNG, not of weight init, not of a model's forward. A torch upgrade that
  perturbs convolution output cannot make this file lie.
* **Both ``ssim_impl`` values are pinned.** ``wang`` and ``daala`` share the
  ``ssim/...`` tag names, so the *only* thing distinguishing them is the value.
  Pinning one convention would leave the other free to drift.
"""

import functools

import pytest
import torch

from sisr.models.srresnet.model import SRResNet
from sisr.processors import RGBProcessor
from sisr.training import SREvalConfig, SRLightning, SRTrainingConfig

# Fixed inputs. Seeded once at module import; regenerating with a different seed
# invalidates every value below.
#
# 🚨 The two images carry DELIBERATELY different error magnitudes. Two equally-wrong
# images make per-image-mean PSNR and whole-batch-pooled PSNR agree exactly, so a
# fixture built from plain `torch.rand` pairs pins nothing about the reduction —
# verified: it lets `dim=(1, 2, 3) -> dim=None` pass untouched. Here the two
# reductions sit 11.35 dB apart, which is what makes them distinguishable.
_g = torch.Generator().manual_seed(20260816)
HR = torch.rand(2, 3, 24, 24, generator=_g)
_noise = torch.rand(2, 3, 24, 24, generator=_g)
SR = HR.clone()
SR[0] = (HR[0] + 0.02 * (_noise[0] - 0.5)).clamp(0, 1)  # barely wrong
SR[1] = (HR[1] + 0.60 * (_noise[1] - 0.5)).clamp(0, 1)  # badly wrong

# Deliberately broad: every psnr_channels form (aggregate, YCbCr, bare channel),
# separate_psnr on, and a non-zero crop_border, so the contract covers the whole
# tag surface rather than the two tags a default config happens to emit.
PSNR_CHANNELS = ["RGB", "YCbCr", "Y"]
SSIM_CHANNELS = ["RGB", "Y"]
CROP_BORDER = 4

# PSNR is independent of ssim_impl by construction; asserting that is the point
# of scoring it under both rather than sharing one copy.
GOLDEN: dict[str, dict[str, float]] = {
    "wang": {
        "psnr/val/R": 30.38504028,
        "psnr/val/G": 30.43302727,
        "psnr/val/B": 30.49940491,
        "psnr/val/RGB": 30.43884277,
        "psnr/val/Y": 35.31136703,
        "psnr/val/Cb": 35.89144135,
        "psnr/val/Cr": 35.01395416,
        "psnr/val/YCbCr": 35.38640594,
        "ssim/val/RGB": 0.9286418557,
        "ssim/val/Y": 0.9239290357,
    },
    "daala": {
        "psnr/val/R": 30.38504028,
        "psnr/val/G": 30.43302727,
        "psnr/val/B": 30.49940491,
        "psnr/val/RGB": 30.43884277,
        "psnr/val/Y": 35.31136703,
        "psnr/val/Cb": 35.89144135,
        "psnr/val/Cr": 35.01395416,
        "psnr/val/YCbCr": 35.38640594,
        "ssim/val/RGB": 0.9231098537,
        "ssim/val/Y": 0.9885816613,
    },
}


def score(ssim_impl: str) -> dict[str, float]:
    """Run ``validation_step`` on fixed tensors and return every logged tag.

    ``_step`` is stubbed rather than mocked at the model level so the crop,
    the colorspace conversions, both reductions and the tag construction all
    run for real — those are the parts a scoring-module refactor would move.
    """
    eval_config = SREvalConfig(
        psnr_channels=PSNR_CHANNELS,
        separate_psnr=True,
        ssim_channels=SSIM_CHANNELS,
        ssim_impl=ssim_impl,
        crop_border=CROP_BORDER,
    )
    module = SRLightning(
        model=SRResNet(scale=4, num_residual_blocks=1),
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=4),
        eval_config=eval_config,
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )

    logged: dict[str, float] = {}

    def fake_log(name, value, **kwargs):
        logged[name] = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)

    module.log = fake_log
    module._step = lambda batch: (torch.tensor(0.0), None, None, SR.clone(), HR.clone())
    module.validation_step((torch.rand(2, 3, 6, 6), HR.clone()), 0)
    return logged


@pytest.mark.parametrize("ssim_impl", ["wang", "daala"])
def test_validation_emits_exactly_the_expected_tags(ssim_impl: str):
    """The tag set is the contract: no additions, no renames, no silent drops.

    ``psnr/val/{key}`` is constructed in three separate places in the codebase
    and ``psnr/{set}/{key}`` in a fourth; set equality here is what makes moving
    them into one owner a refactor rather than a rename.
    """
    logged = score(ssim_impl)
    assert set(logged) == set(GOLDEN[ssim_impl]) | {"loss/val"}


@pytest.mark.parametrize("ssim_impl", ["wang", "daala"])
def test_validation_values_are_pinned(ssim_impl: str):
    """Every scored number, to 9 significant figures, on fixed inputs."""
    logged = score(ssim_impl)
    for tag, expected in GOLDEN[ssim_impl].items():
        assert logged[tag] == pytest.approx(expected, rel=1e-9), tag


def test_psnr_is_identical_across_ssim_impl():
    """`ssim_impl` must not reach the PSNR path.

    Without this, a refactor that threaded the impl through a shared scorer
    could perturb PSNR and still satisfy both pinned tables above, because each
    table was regenerated from the same broken code.
    """
    wang, daala = score("wang"), score("daala")
    psnr_tags = [t for t in wang if t.startswith("psnr/")]
    assert psnr_tags
    for tag in psnr_tags:
        assert wang[tag] == daala[tag], tag


def test_the_two_ssim_conventions_genuinely_differ():
    """Guards the pinning against vacuity.

    If `ssim_impl` silently stopped being honoured, both tables above would
    still pass once regenerated — they would just be the same numbers twice.
    Trap 2 in miniature: same tag, different meaning, no way to tell them apart
    after the fact.
    """
    wang, daala = score("wang"), score("daala")
    for tag in (t for t in wang if t.startswith("ssim/")):
        assert wang[tag] != pytest.approx(daala[tag], rel=1e-6), tag


# --- crop_border: the two failures that reach a number, not an exception ---


def test_crop_rejects_a_negative_border_forced_past_config_validation():
    """SRLightning ASSIGNS eval_config.crop_border after construction, which
    bypasses __post_init__ entirely -- so config validation alone cannot be the
    only guard. A negative border makes the slice guard `n <= 0` return the
    tensors uncropped: a number that looks valid and is comparable to nothing."""
    from sisr.metrics.scoring import SRScorer
    from sisr.training import SREvalConfig

    cfg = SREvalConfig(crop_border=0)
    cfg.crop_border = -4  # what _resolved_scale would have written for scale=-4
    scorer = SRScorer(cfg)
    with pytest.raises(ValueError, match="crop_border"):
        scorer.crop(torch.zeros(1, 3, 32, 32))


def test_crop_rejects_a_border_that_would_consume_the_whole_image():
    """Oversized borders currently surface as a bare torch RuntimeError several
    frames down, naming no config key. Every sibling field gets an actionable
    message; this one should too, and it names the image size because that is
    the half of the problem the config cannot know."""
    from sisr.metrics.scoring import SRScorer
    from sisr.training import SREvalConfig

    scorer = SRScorer(SREvalConfig(crop_border=99))
    with pytest.raises(ValueError, match=r"crop_border"):
        scorer.crop(torch.zeros(1, 3, 8, 8))


def test_crop_accepts_a_border_that_leaves_at_least_one_pixel():
    """The boundary stays usable: 3 px off each side of an 8x8 leaves 2x2."""
    from sisr.metrics.scoring import SRScorer
    from sisr.training import SREvalConfig

    (out,) = SRScorer(SREvalConfig(crop_border=3)).crop(torch.zeros(1, 3, 8, 8))
    assert out.shape == (1, 3, 2, 2)
