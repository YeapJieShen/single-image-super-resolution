"""No-reference perceptual metrics (NIQE, Ma, PI) as the scorer sees them.

NIQE is the native port (:mod:`sisr.metrics.niqe`); Ma comes from pyiqa. The kwarg
and dispatch tests stub the ``_pyiqa_metric`` seam and inject a synthetic pristine
model, so they need neither pyiqa, SciPy nor any download and run in CI.
"""

import subprocess
import sys

import pytest
import torch

from sisr.metrics import niqe as niqe_mod
from sisr.metrics import perceptual
from sisr.metrics.perceptual import (
    NO_REFERENCE_METRICS,
    PERCEPTUAL_METRICS,
    no_reference_score,
    perceptual_score,
)


class _MaSpy(torch.nn.Module):
    """Stands in for pyiqa's nrqm: one fixed score per image, unreduced."""

    calls: list = []

    def __init__(self, name, **kwargs):
        super().__init__()
        _MaSpy.calls.append((name, kwargs))

    def forward(self, x):
        assert x.device == torch.device("cpu") and x.dtype == torch.float32
        return torch.tensor([1.0, 3.0])[: x.shape[0]]


@pytest.fixture
def ma_spy(monkeypatch):
    _MaSpy.calls = []
    monkeypatch.setattr(perceptual, "_NET_CACHE", {})
    monkeypatch.setattr(perceptual, "_pyiqa_metric", lambda name, **kw: _MaSpy(name, **kw))


@pytest.fixture
def synthetic_model(monkeypatch):
    """The real pristine model needs SciPy and a download; the pipeline does not."""
    model = (torch.zeros(36, dtype=torch.float64), torch.eye(36, dtype=torch.float64))
    monkeypatch.setattr(niqe_mod, "pristine_model", lambda: model)
    return model


def _images(n: int = 2, size: int = 104) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    base = torch.rand(n, 3, size // 8, size // 8, generator=g)
    smooth = torch.nn.functional.interpolate(base, size=(size, size), mode="bilinear")
    return (smooth + 0.1 * torch.rand(n, 3, size, size, generator=g)).clamp(0, 1)


# Spelled out here on purpose, separately from the module constant: a test that
# imported the constant would pass whatever the constant said.
_EXPECTED_MA = {
    "ma": dict(test_y_channel=True, color_space="yiq", crop_border=0),
    "pi": dict(test_y_channel=True, color_space="ycbcr", crop_border=4),
}


@pytest.mark.parametrize("name", ["ma", "pi"])
def test_ma_kwargs_are_pinned_explicitly(ma_spy, synthetic_model, name):
    """Every convention kwarg is passed, none left at pyiqa's defaults, and Ma is
    built on the CPU (its random forest is CPU-only)."""
    no_reference_score(name, _images())

    ((pyiqa_name, kwargs),) = _MaSpy.calls
    assert pyiqa_name == "nrqm"
    assert {k: v for k, v in kwargs.items() if k != "device"} == _EXPECTED_MA[name]
    assert kwargs["device"] == torch.device("cpu")


def test_niqe_needs_no_pyiqa(ma_spy, synthetic_model):
    no_reference_score("niqe", _images())

    assert _MaSpy.calls == []


def test_niqe_is_the_native_port_batch_meaned(synthetic_model):
    sr = _images()

    expected = niqe_mod.niqe(sr, model=synthetic_model).mean()

    assert no_reference_score("niqe", sr).item() == pytest.approx(expected.item())


def test_pi_is_half_of_ten_minus_ma_plus_niqe_at_pirm_conventions(ma_spy, synthetic_model):
    """PIRM's definition on Ma's 0-10 scale (not 100 - Ma), with NIQE on PIRM's Y
    and 4-pixel shave -- the same convention as PI's Ma term."""
    sr = _images()
    niqe_y = niqe_mod.niqe(sr, luma="ycbcr", shave=4, model=synthetic_model)
    ma = torch.tensor([1.0, 3.0], dtype=torch.float64)

    expected = (0.5 * ((10 - ma) + niqe_y)).mean()

    assert no_reference_score("pi", sr).item() == pytest.approx(expected.item())


def test_ma_reduces_to_the_batch_mean_on_the_input_device_and_dtype(ma_spy):
    sr = _images().to(torch.float64)

    score = no_reference_score("ma", sr)

    assert score.ndim == 0 and score.dtype == torch.float64
    assert score.item() == pytest.approx(2.0)


def test_table_and_directions():
    assert NO_REFERENCE_METRICS == {"niqe", "ma", "pi"}
    assert PERCEPTUAL_METRICS["niqe"] is True
    assert PERCEPTUAL_METRICS["pi"] is True
    assert PERCEPTUAL_METRICS["ma"] is False  # Ma is higher-better
    assert PERCEPTUAL_METRICS["lpips"] is True and PERCEPTUAL_METRICS["dists"] is True


def test_hr_is_ignored_by_no_reference_names(ma_spy, synthetic_model):
    sr = _images()

    for name in sorted(NO_REFERENCE_METRICS):
        with_hr = perceptual_score(name, sr, torch.rand_like(sr))
        without = perceptual_score(name, sr)
        assert with_hr.item() == pytest.approx(without.item())


@pytest.mark.parametrize("name", ["lpips", "dists"])
def test_full_reference_names_reject_missing_hr(name):
    with pytest.raises(ValueError, match=name):
        perceptual_score(name, torch.rand(1, 3, 32, 32), None)


def test_no_reference_score_rejects_full_reference_names():
    with pytest.raises(ValueError, match="lpips"):
        no_reference_score("lpips", torch.rand(1, 3, 32, 32))


def test_ma_is_built_once_per_convention(ma_spy, synthetic_model):
    sr = _images()
    for _ in range(2):
        no_reference_score("ma", sr)
        no_reference_score("pi", sr)

    assert [kw["color_space"] for _, kw in _MaSpy.calls] == ["yiq", "ycbcr"]


def test_missing_pyiqa_is_an_actionable_import_error(monkeypatch):
    monkeypatch.setattr(perceptual, "_NET_CACHE", {})
    monkeypatch.setitem(sys.modules, "pyiqa", None)

    with pytest.raises(ImportError, match=r"pip install '\.\[perceptual\]'"):
        no_reference_score("ma", torch.rand(1, 3, 32, 32))


def test_importing_the_module_does_not_import_pyiqa_or_scipy():
    code = (
        "import sys, sisr.metrics.perceptual; "
        "sys.exit(any(m in sys.modules for m in ('pyiqa', 'scipy')))"
    )

    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_eval_config_accepts_no_reference_names():
    from sisr.training.config import SREvalConfig

    SREvalConfig(perceptual_metrics=["niqe", "ma", "pi"])


# --- behaviour lock: needs pyiqa and its weights --------------------------------------


def test_pinned_ma_matches_a_direct_pyiqa_call():
    pyiqa = pytest.importorskip("pyiqa")
    sr = _images(1, 128)

    for name, kwargs in _EXPECTED_MA.items():
        direct = float(pyiqa.create_metric("nrqm", device="cpu", **kwargs)(sr))
        assert float(perceptual._ma(sr, name)[0]) == pytest.approx(direct)


def test_scorer_scores_no_reference_names_on_the_cropped_reconstruction(synthetic_model):
    """The scorer hands NR metrics the same crop_border-cropped SR the paired
    metrics see, and never needs the HR for them."""
    from sisr.metrics.scoring import SRScorer
    from sisr.training.config import SREvalConfig

    sr = _images(1, 112)
    scorer = SRScorer(SREvalConfig(crop_border=4, perceptual_metrics=["niqe"]))

    scores = scorer.score(sr, torch.rand_like(sr))

    expected = no_reference_score("niqe", sr[..., 4:-4, 4:-4])
    assert scores.perceptual["niqe"].item() == pytest.approx(expected.item())
    assert scores.perceptual["niqe"].item() != pytest.approx(no_reference_score("niqe", sr).item())
