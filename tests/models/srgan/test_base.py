"""Contract tests for AdversarialDiscriminator — the abstract base other discriminators subclass."""

import pytest
import torch.nn as nn

from sisr.models.srgan import AdversarialDiscriminator, SRDiscriminator


def test_adversarial_discriminator_is_abstract():
    """Cannot be instantiated directly — four members are abstract."""
    with pytest.raises(TypeError, match="abstract"):
        AdversarialDiscriminator()


def test_subclass_inherits_nn_module():
    class _Trivial(AdversarialDiscriminator):
        def __init__(self):
            super().__init__()
            self._hparams = {"in_channels": 3}
            self.conv = nn.Conv2d(3, 1, 1)

        @property
        def in_channels(self) -> int:
            return self._hparams["in_channels"]

        @property
        def input_size(self) -> int | None:
            return None

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
        def in_channels(self) -> int:
            return 3

        @property
        def input_size(self) -> int | None:
            return None

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

        @property
        def in_channels(self) -> int:
            return 3

        @property
        def input_size(self) -> int | None:
            return None

        def forward(self, x):
            return x

    with pytest.raises(TypeError, match="variant_tag"):
        _NoTag()


def test_refuses_a_subclass_with_no_forward():
    """forward is abstract on purpose — a subclass must define its own architecture."""

    class _NoForward(AdversarialDiscriminator):
        def __init__(self):
            super().__init__()
            self._hparams = {}

        @property
        def in_channels(self) -> int:
            return 3

        @property
        def input_size(self) -> int | None:
            return None

        @property
        def variant_tag(self) -> str:
            return "none"

    with pytest.raises(TypeError, match="forward"):
        _NoForward()


def test_refuses_a_subclass_with_no_in_channels():
    """in_channels is abstract on purpose — it gates construction validation."""

    class _NoInChannels(AdversarialDiscriminator):
        def __init__(self):
            super().__init__()
            self._hparams = {}

        @property
        def input_size(self) -> int | None:
            return None

        @property
        def variant_tag(self) -> str:
            return "none"

        def forward(self, x):
            return x

    with pytest.raises(TypeError, match="in_channels"):
        _NoInChannels()


def test_refuses_a_subclass_with_no_input_size():
    """input_size is abstract on purpose — it gates setup-time validation."""

    class _NoInputSize(AdversarialDiscriminator):
        def __init__(self):
            super().__init__()
            self._hparams = {}

        @property
        def in_channels(self) -> int:
            return 3

        @property
        def variant_tag(self) -> str:
            return "none"

        def forward(self, x):
            return x

    with pytest.raises(TypeError, match="input_size"):
        _NoInputSize()


def test_srdiscriminator_is_a_subclass():
    assert issubclass(SRDiscriminator, AdversarialDiscriminator)
