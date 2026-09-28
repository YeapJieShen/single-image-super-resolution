"""Abstract base for discriminators in an adversarially-trained SR pipeline."""

import abc

import torch
import torch.nn as nn


class AdversarialDiscriminator(nn.Module, abc.ABC):
    """Abstract base for the critic paired with a generator under adversarial training.

    Deliberately **not** an :class:`~sisr.models.base.SRModel`: that contract
    declares ``input_contract`` and ``reset_parameters``, both meaningless for
    a classifier -- inheriting it would mean declaring a contract that is a
    lie. This base exists so a second discriminator architecture is validated
    against a shared, minimal contract instead of being duck-typed against
    :class:`~sisr.models.srgan.SRDiscriminator` specifically.

    Subclasses must populate ``self._hparams`` in ``__init__``, implement
    ``forward``, and declare ``variant_tag``.
    """

    _hparams: dict

    @property
    def hparams(self) -> dict:
        """Architecture hyperparameters dict, for provenance metadata."""
        return self._hparams

    @property
    @abc.abstractmethod
    def variant_tag(self) -> str:
        """Short token distinguishing this configuration from siblings of the same class.

        Abstract for the same reason ``SRModel.variant_tag`` is: no rule over
        ``hparams`` yields a readable tag for every architecture, and an
        inherited default would label two configurations identically in a
        weights filename.
        """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch and return **logits** (not probabilities), shape ``(B, 1)``."""
