"""Abstract base class for single-image super-resolution architectures."""

import abc
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn


class SRModel(nn.Module, abc.ABC):
    """Abstract base for SR architectures.

    Subclasses must populate ``self._hparams`` in ``__init__``, implement
    ``forward``, and declare ``input_contract``. ``reset_parameters`` is a
    no-op by default; override it for paper-faithful weight init schemes.
    """

    _hparams: dict

    #: How the model expects its LR input: ``'pre_upsampled'`` if LR arrives
    #: already resized to the HR grid (e.g. SRCNN, which is resolution-
    #: preserving), or ``'native_lr'`` if the model consumes true low-
    #: resolution and upsamples internally (e.g. SRResNet's sub-pixel conv).
    #: Declared per-architecture rather than inferred (e.g. from ``'scale'
    #: in hparams``) — that check happens to hold for both current
    #: architectures but would silently break on a third that doesn't follow
    #: the same pattern.
    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]]

    @property
    def hparams(self) -> dict:
        """Architecture hyperparameters dict for the Lightning HParams merge."""
        return self._hparams

    @property
    @abc.abstractmethod
    def variant_tag(self) -> str:
        """Short token distinguishing this configuration from siblings of the same class.

        Appears in artifact filenames, so a directory of weights can be read
        without opening anything: ``SRCNN_x2_Y_915`` and ``SRResNet_x4_RGB_16B64F``
        differ in exactly this token when the architecture and scale match.

        Abstract, never defaulted, for the same reason ``input_contract`` is
        declared rather than inferred: no rule over ``hparams`` produces a
        readable tag for every architecture, and an inherited default would
        silently label two different configurations identically. Keep it short,
        stable, and filename-safe.
        """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model on input ``x`` and return the SR output tensor."""

    def reset_parameters(self, **kwargs: Any) -> None:
        """Optional paper-style weight init. Default: no-op (kwargs ignored).

        Subclasses may declare specific kwargs (e.g. ``SRCNN``'s ``mean`` /
        ``std``). The base accepts ``**kwargs`` so callers like
        ``SRLightning`` can pass paper-init options
        polymorphically without knowing the subclass signature — models that
        don't override absorb the kwargs as no-ops.
        """
        pass
