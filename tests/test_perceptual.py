import pytest
import torch

from sisr.perceptual import PERCEPTUAL_METRICS, perceptual_score


def test_lpips_receives_normalize_true(monkeypatch):
    """[0,1] inputs must be declared as such to LPIPS.

    torchmetrics defaults normalize=False, which means "inputs are already
    [-1,1]". Our metric tensors are RGB [0,1], so the default silently scores
    a squashed range: a plausible number, no error, uncomparable to anything.
    """
    seen = {}

    def spy(img1, img2, net_type, reduction, normalize):
        seen.update(net_type=net_type, reduction=reduction, normalize=normalize)
        return torch.tensor(0.25)

    monkeypatch.setattr("sisr.perceptual._lpips_fn", lambda: spy)
    perceptual_score("lpips", torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))

    assert seen["normalize"] is True
    assert seen["reduction"] == "mean"
    assert seen["net_type"] == "alex"


def test_dists_reduction_is_passed_explicitly(monkeypatch):
    """DISTS defaults reduction=None (per-image), LPIPS defaults 'mean'.

    Relying on either default gives a scalar from one metric and a tensor from
    the other under the same self.log call.
    """
    seen = {}

    def spy(preds, target, reduction):
        seen["reduction"] = reduction
        return torch.tensor(0.1)

    monkeypatch.setattr("sisr.perceptual._dists_fn", lambda: spy)
    perceptual_score("dists", torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))

    assert seen["reduction"] == "mean"


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
    monkeypatch.setattr("torchmetrics.utilities.imports._LPIPS_AVAILABLE", False)

    with pytest.raises(ImportError, match=r"pip install '\.\[perceptual\]'"):
        perceptual_score("lpips", torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
