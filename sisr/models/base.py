"""Abstract base class for single-image super-resolution architectures."""

import abc
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn


def as_int_tuple(name: str, value: Sequence[int]) -> tuple[int, ...]:
    """Normalise a per-layer sequence hyperparameter to a tuple.

    Provenance metadata records hparams tuple-free (JSON has no tuple), so a
    list must rebuild the same model a tuple built. Element values are left to
    the caller's own checks.

    Args:
        name: The parameter's name, for the error message.
        value: A list or tuple.

    Returns:
        ``value`` as a tuple.

    Raises:
        ValueError: If ``value`` is a ``str`` (a sequence, but of characters)
            or is not a sequence at all.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list or tuple. Got {type(value)}.")
    return tuple(value)


class SRModel(nn.Module, abc.ABC):
    """Abstract base for SR architectures.

    Subclasses must populate ``self._hparams`` in ``__init__``, implement
    ``forward``, and declare ``input_contract``. ``reset_parameters`` is a
    no-op by default; override it for paper-faithful weight init schemes.
    """

    _hparams: dict

    #: How the model expects its LR input: ``'pre_upsampled'`` if LR arrives
    #: already on the HR grid (SRCNN), ``'native_lr'`` if the model upsamples
    #: internally (SRResNet). Declared, never inferred -- a rule like
    #: ``'scale' in hparams`` holds for both current architectures and would
    #: silently break on a third.
    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]]

    @property
    def hparams(self) -> dict:
        """Architecture hyperparameters dict for the Lightning HParams merge."""
        return self._hparams

    @property
    @abc.abstractmethod
    def variant_tag(self) -> str:
        """Short token distinguishing this configuration from siblings of the same class.

        Appears in artifact filenames, so a directory of weights reads without
        opening anything: ``SRCNN_x2_Y_915``, ``SRResNet_x4_RGB_16B64F``.

        Abstract, never defaulted, for the same reason ``input_contract`` is:
        no rule over ``hparams`` yields a readable tag for every architecture,
        and an inherited default would label two configurations identically.
        Keep it short, stable and filename-safe.
        """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model on input ``x`` and return the SR output tensor."""

    def reset_parameters(self, **kwargs: Any) -> None:
        """Optional paper-style weight init. Default: no-op.

        ``**kwargs`` lets ``SRLightning`` pass paper-init options
        polymorphically; subclasses declare what they read (``SRCNN``'s
        ``mean``/``std``), and models that do not override absorb them.
        """
        pass
