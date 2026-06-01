"""Contract tests for SRModel — the abstract base for SR architectures."""
import pytest
import torch
import torch.nn as nn

from sisr.models.base import SRModel


def test_srmodel_is_abstract():
    """SRModel cannot be instantiated directly — forward is abstract."""
    with pytest.raises(TypeError, match="abstract"):
        SRModel()


def test_srmodel_subclass_inherits_nn_module():
    """A concrete SRModel subclass is also an nn.Module (for Lightning interop)."""
    class _Trivial(SRModel):
        def __init__(self):
            super().__init__()
            self._hparams = {"trivial": True}
            self.conv = nn.Conv2d(3, 3, 1)
        def forward(self, x): return self.conv(x)

    m = _Trivial()
    assert isinstance(m, nn.Module)
    assert isinstance(m, SRModel)


def test_srmodel_hparams_returns_underlying_dict():
    """hparams property exposes self._hparams as a read-only view."""
    class _Trivial(SRModel):
        def __init__(self):
            super().__init__()
            self._hparams = {"foo": 1, "bar": "two"}
        def forward(self, x): return x

    m = _Trivial()
    assert m.hparams == {"foo": 1, "bar": "two"}


def test_srmodel_reset_parameters_default_is_noop():
    """Default reset_parameters returns None, accepts arbitrary kwargs, and does not mutate parameters.

    The ``**kwargs`` signature lets SRLightning pass paper-init kwargs (e.g. ``mean``,
    ``std`` from SRCNN) polymorphically without knowing the subclass signature —
    base subclasses without paper init absorb the kwargs as no-ops.
    """
    class _Trivial(SRModel):
        def __init__(self):
            super().__init__()
            self._hparams = {}
            self.conv = nn.Conv2d(3, 3, 1)
        def forward(self, x): return self.conv(x)

    m = _Trivial()
    before = m.conv.weight.detach().clone()
    assert m.reset_parameters() is None                       # no kwargs
    assert m.reset_parameters(mean=0.5, std=0.1) is None      # kwargs absorbed
    assert torch.equal(m.conv.weight, before)
