"""Abstract base class for single-image super-resolution architectures."""

import abc

import torch
import torch.nn as nn


class SRModel(nn.Module, abc.ABC):
    """Abstract base for SR architectures.

    Subclasses must populate ``self._hparams`` in ``__init__`` and
    implement :meth:`forward`. ``reset_parameters`` is a no-op by default;
    override it for paper-faithful weight init schemes.
    """

    _hparams: dict

    @property
    def hparams(self) -> dict:
        """Architecture hyperparameters dict for the Lightning HParams merge."""
        return self._hparams

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model on input ``x`` and return the SR output tensor."""

    def reset_parameters(self, **kwargs) -> None:
        """Optional paper-style weight init. Default: no-op (kwargs ignored).

        Subclasses may declare specific kwargs (e.g. :class:`SRCNN`'s ``mean`` /
        ``std``). The base accepts ``**kwargs`` so callers like
        :class:`~sisr.training.SRLightning` can pass paper-init options
        polymorphically without knowing the subclass signature — models that
        don't override absorb the kwargs as no-ops.
        """
        pass
