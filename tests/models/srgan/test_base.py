"""Contract tests for AdversarialDiscriminator — the abstract base other discriminators subclass."""

import pytest
import torch.nn as nn

from sisr.models.srgan import AdversarialDiscriminator, SRDiscriminator


def test_adversarial_discriminator_is_abstract():
    """Cannot be instantiated directly — forward and variant_tag are abstract."""
    with pytest.raises(TypeError, match="abstract"):
        AdversarialDiscriminator()


def test_subclass_inherits_nn_module():
    class _Trivial(AdversarialDiscriminator):
        def __init__(self):
            super().__init__()
            self._hparams = {"in_channels": 3}
            self.conv = nn.Conv2d(3, 1, 1)

        @property
        def variant_tag(self):
            return "t"

        def forward(self, x):
            return self.conv(x).mean(dim=(1, 2, 3), keepdim=True)

    d = _Trivial()
    assert isinstance(d, nn.Module)
    assert isinstance(d, AdversarialDiscriminator)


def test_hparams_returns_underlying_dict():
    class _Trivial(AdversarialDiscriminator):
        def __init__(self):
            super().__init__()
            self._hparams = {"foo": 1, "bar": "two"}

        @property
        def variant_tag(self):
            return "t"

        def forward(self, x):
            return x

    assert _Trivial().hparams == {"foo": 1, "bar": "two"}


def test_refuses_a_subclass_with_no_variant_tag():
    """variant_tag is abstract on purpose — see SRModel's identical test."""

    class _NoTag(AdversarialDiscriminator):
        def __init__(self):
            super().__init__()
            self._hparams = {}

        def forward(self, x):
            return x

    with pytest.raises(TypeError, match="variant_tag"):
        _NoTag()


def test_srdiscriminator_is_a_subclass():
    assert issubclass(SRDiscriminator, AdversarialDiscriminator)
