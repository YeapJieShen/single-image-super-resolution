"""Contract tests for SRProcessor — the abstract base for colorspace adapters."""

import pytest
import torch

from sisr.processors.base import SRProcessor


def test_srprocessor_is_abstract():
    """SRProcessor cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        SRProcessor()


def test_srprocessor_subclass_must_implement_both_methods():
    """A subclass missing either abstract method also raises on instantiation."""

    class _ExtractOnly(SRProcessor):
        def extract(self, lr_rgb):
            return lr_rgb

    with pytest.raises(TypeError, match="abstract"):
        _ExtractOnly()

    class _ReconstructOnly(SRProcessor):
        def reconstruct(self, sr_model_out, lr_rgb):
            return sr_model_out

    with pytest.raises(TypeError, match="abstract"):
        _ReconstructOnly()


def test_srprocessor_complete_subclass_instantiates():
    """Subclass implementing both abstract methods can be instantiated."""

    class _Complete(SRProcessor):
        def extract(self, lr_rgb):
            return lr_rgb

        def reconstruct(self, sr_model_out, lr_rgb):
            return sr_model_out

    p = _Complete()
    assert isinstance(p, SRProcessor)
    x = torch.zeros(1, 3, 4, 4)
    assert torch.equal(p.extract(x), x)
    assert torch.equal(p.reconstruct(x, x), x)
