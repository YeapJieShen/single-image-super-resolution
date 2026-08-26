"""Perceptual-metric tests.

Every test that would otherwise build a backbone monkeypatches the import seam
(``_lpips_metric_cls`` / ``_dists_net_cls``): a real construction downloads
~230 MB of AlexNet or ~528 MB of VGG16 into the torch hub cache, which CI has
neither the network nor the cache for. ``_NET_CACHE`` is swapped for a fresh
dict alongside, so a spy from one test can never be served to another.
"""

import pytest
import torch

from sisr.metrics.perceptual import PERCEPTUAL_METRICS, perceptual_score


class _LpipsSpy(torch.nn.Module):
    """Stands in for ``LearnedPerceptualImagePatchSimilarity``.

    Records its construction kwargs (the metric's two conventions live there
    now, not at the call site) and whether the per-call state reset fired.
    """

    seen: dict = {}

    def __init__(self, **kwargs):
        super().__init__()
        _LpipsSpy.seen = {**kwargs, "constructions": _LpipsSpy.seen.get("constructions", 0) + 1}
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, img1, img2):
        return torch.tensor(0.25)

    def reset(self):
        _LpipsSpy.seen["reset"] = _LpipsSpy.seen.get("reset", 0) + 1


class _DistsSpy(torch.nn.Module):
    """Stands in for ``DISTSNetwork`` — returns one score per image, unreduced."""

    constructions = 0

    def __init__(self):
        super().__init__()
        _DistsSpy.constructions += 1
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, y, require_grad=False):
        return torch.tensor([0.1, 0.3])


@pytest.fixture
def spies(monkeypatch):
    """Both import seams stubbed, on a cache private to this test."""
    _LpipsSpy.seen = {}
    _DistsSpy.constructions = 0
    monkeypatch.setattr("sisr.metrics.perceptual._NET_CACHE", {})
    monkeypatch.setattr("sisr.metrics.perceptual._lpips_metric_cls", lambda: _LpipsSpy)
    monkeypatch.setattr("sisr.metrics.perceptual._dists_net_cls", lambda: _DistsSpy)


@pytest.mark.parametrize("net", ["alex", "vgg", "squeeze"])
def test_lpips_backbone_is_threaded_through(spies, net):
    """`lpips_net` must reach the backbone, and a LPIPS figure is comparable only
    to one computed under the same backbone — a hardcoded 'alex' would be a
    silently uncomparable number, not an error."""
    perceptual_score("lpips", torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32), lpips_net=net)

    assert _LpipsSpy.seen["net_type"] == net


def test_lpips_receives_normalize_true(spies):
    """[0,1] inputs must be declared as such to LPIPS.

    torchmetrics defaults normalize=False, which means "inputs are already
    [-1,1]". Our metric tensors are RGB [0,1], so the default silently scores
    a squashed range: a plausible number, no error, uncomparable to anything.

    Passed at construction rather than per call now that the backbone is
    memoised — the modular metric takes both conventions in ``__init__``.
    """
    perceptual_score("lpips", torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))

    assert _LpipsSpy.seen["normalize"] is True
    assert _LpipsSpy.seen["reduction"] == "mean"


def test_dists_is_reduced_to_a_batch_mean(spies):
    """DISTS's own default reduction is None (per-image), LPIPS's is 'mean'.

    Relying on either default gives a scalar from one metric and a tensor from
    the other under the same self.log call. The reduction is applied here rather
    than passed as an argument now — the cached backbone is the bare network,
    below torchmetrics' reduction layer — so this asserts the resulting shape and
    value instead of the argument.
    """
    score = perceptual_score("dists", torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32))

    assert score.ndim == 0
    assert score.item() == pytest.approx(0.2)


@pytest.mark.parametrize("name", ["lpips", "dists"])
def test_backbone_is_built_once_across_calls(spies, name):
    """The whole cost of these metrics was rebuilding the backbone per call.

    Measured on CPU at 1x3x256x256: 605.0 -> 27.3 ms/call for LPIPS and
    1716.1 -> 386.8 ms/call for DISTS. One validation cycle of the shipped
    adversarial template scores ~219 images per metric.
    """
    for _ in range(3):
        perceptual_score(name, torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))

    built = _LpipsSpy.seen["constructions"] if name == "lpips" else _DistsSpy.constructions
    assert built == 1


def test_each_lpips_backbone_gets_its_own_cache_entry(spies):
    """Caching on the metric name alone would serve an 'alex' backbone to a run
    that asked for 'vgg' — the same uncomparable-number failure, cached."""
    for net in ("alex", "vgg", "alex"):
        perceptual_score("lpips", torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32), lpips_net=net)

    assert _LpipsSpy.seen["constructions"] == 2


def test_lpips_state_is_reset_after_every_call(spies):
    """The modular metric accumulates a list state per update. Reused across a
    400k-step run without a reset, it retains one tensor per scored image."""
    perceptual_score("lpips", torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))

    assert _LpipsSpy.seen["reset"] == 1


@pytest.mark.parametrize("name", ["lpips", "dists"])
def test_cached_backbone_cannot_train_or_leak_gradients(spies, name):
    """A memoised backbone outlives the call that built it, so it must be inert:
    eval mode, detached parameters, and no graph reaching the scored tensors."""
    from sisr.metrics.perceptual import _cached_net

    sr = torch.rand(1, 3, 32, 32, requires_grad=True)
    score = perceptual_score(name, sr, torch.rand(1, 3, 32, 32))
    net = _cached_net(name, "alex", sr.device, sr.dtype)

    assert not net.training
    assert not any(p.requires_grad for p in net.parameters())
    assert not score.requires_grad
    assert score.grad_fn is None


def test_unknown_metric_names_the_supported_set():
    with pytest.raises(ValueError, match="lpips"):
        perceptual_score("ssim", torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))


def test_every_metric_declares_a_direction():
    """A metric with no declared direction would be monitored with the wrong mode."""
    assert PERCEPTUAL_METRICS == {"lpips": True, "dists": True}


def test_missing_lpips_extra_names_the_install_command(monkeypatch):
    """A missing `lpips` package must fail with the install command, not a bare error.

    torchmetrics gates LPIPS behind `_LPIPS_AVAILABLE`; regressing to a generic
    ImportError (or any error that drops the extra's name) would leave anyone
    hitting this with no way to know which package fixes it.
    """
    monkeypatch.setattr("sisr.metrics.perceptual._NET_CACHE", {})
    monkeypatch.setattr("torchmetrics.utilities.imports._LPIPS_AVAILABLE", False)

    with pytest.raises(ImportError, match=r"pip install '\.\[perceptual\]'"):
        perceptual_score("lpips", torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
