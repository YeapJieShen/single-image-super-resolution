"""Contract tests for SRProcessor — the abstract base for colorspace adapters."""

import pytest
import torch

from sisr.processors.base import SRProcessor


def test_srprocessor_is_abstract():
    """SRProcessor cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        SRProcessor()


def test_srprocessor_subclass_must_implement_all_abstract_members():
    """A subclass missing any abstract member also raises on instantiation."""

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


def test_srprocessor_subclass_missing_model_channels_raises():
    """model_channels is abstract too — extract/reconstruct alone isn't enough."""

    class _MissingModelChannels(SRProcessor):
        def extract(self, lr_rgb):
            return lr_rgb

        def reconstruct(self, sr_model_out, lr_rgb):
            return sr_model_out

        @property
        def output_range(self):
            return (0.0, 1.0)

    with pytest.raises(TypeError, match="abstract"):
        _MissingModelChannels()


def test_srprocessor_subclass_missing_output_range_raises():
    """output_range is abstract too — model_channels alone isn't enough."""

    class _MissingOutputRange(SRProcessor):
        def extract(self, lr_rgb):
            return lr_rgb

        def reconstruct(self, sr_model_out, lr_rgb):
            return sr_model_out

        @property
        def model_channels(self):
            return 3

    with pytest.raises(TypeError, match="abstract"):
        _MissingOutputRange()


def test_srprocessor_subclass_missing_output_colorspace_raises():
    """output_colorspace is abstract too — model_channels/output_range alone isn't enough."""

    class _MissingOutputColorspace(SRProcessor):
        def extract(self, lr_rgb):
            return lr_rgb

        def reconstruct(self, sr_model_out, lr_rgb):
            return sr_model_out

        @property
        def model_channels(self):
            return 3

        @property
        def output_range(self):
            return (0.0, 1.0)

    with pytest.raises(TypeError, match="abstract"):
        _MissingOutputColorspace()


def test_srprocessor_complete_subclass_instantiates():
    """Subclass implementing all abstract members can be instantiated."""

    class _Complete(SRProcessor):
        def extract(self, lr_rgb):
            return lr_rgb

        def reconstruct(self, sr_model_out, lr_rgb):
            return sr_model_out

        @property
        def model_channels(self):
            return 3

        @property
        def output_range(self):
            return (0.0, 1.0)

        @property
        def output_colorspace(self):
            return "RGB"

    p = _Complete()
    assert isinstance(p, SRProcessor)
    x = torch.zeros(1, 3, 4, 4)
    assert torch.equal(p.extract(x), x)
    assert torch.equal(p.reconstruct(x, x), x)
    assert p.model_channels == 3
    assert p.output_range == (0.0, 1.0)
    assert p.output_colorspace == "RGB"


def test_extract_target_is_concrete_and_delegates_to_extract():
    """extract_target is not abstract — a symmetric processor inherits the right behaviour."""

    class _Doubling(SRProcessor):
        def extract(self, lr_rgb):
            return lr_rgb * 2

        def reconstruct(self, sr_model_out, lr_rgb):
            return sr_model_out / 2

        @property
        def model_channels(self):
            return 3

        @property
        def output_range(self):
            return (0.0, 1.0)

        @property
        def output_colorspace(self):
            return "RGB"

    p = _Doubling()  # instantiates without defining extract_target
    x = torch.rand(1, 3, 4, 4)
    assert torch.equal(p.extract_target(x), p.extract(x))
