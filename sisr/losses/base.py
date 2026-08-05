"""Abstract base for losses that must adapt to the model's output space."""

import abc

import torch

from ..processors import SRProcessor


class SRLoss(torch.nn.Module, abc.ABC):
    """A criterion that needs facts only the :class:`SRProcessor` knows.

    :class:`~sisr.training.lightning_module.SRLightning` calls :meth:`bind`
    exactly once at construction, passing the processor wired alongside the
    model. Plain :class:`torch.nn.Module` criteria (``MSELoss``, ``L1Loss``,
    :class:`~sisr.losses.pixel.CharbonnierLoss`) need no such adaptation and
    are used directly — this base exists only for the losses that do.
    """

    @abc.abstractmethod
    def bind(self, processor: SRProcessor) -> None:
        """Adopt the model's output space, or raise if it cannot be served.

        Abstract rather than defaulted to a no-op, for the same reason
        :attr:`SRProcessor.output_range` is: a wrong inherited default is
        exactly the failure this declaration exists to prevent. It is also
        the one place an impossible model/loss pairing can fail loudly
        instead of silently computing the wrong quantity.

        Args:
            processor: The processor wired alongside the model, supplying
                ``output_range`` and ``model_channels``.

        Raises:
            ValueError: If this loss cannot operate on the processor's
                output space.
        """

    @abc.abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scalar loss between a model output and its target, both in model space."""

    def describe(self) -> str:
        """One-line recipe for the TensorBoard HParams column.

        Concrete with a class-name default: this is presentation, so a new
        loss must not be forced to implement it.
        """
        return type(self).__name__
