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
