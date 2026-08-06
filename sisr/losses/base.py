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

    A loss may optionally expose ``last_terms: dict[str, torch.Tensor]``, a
    structural protocol :class:`SRLightning` checks via ``getattr`` rather
    than ``isinstance``, so any criterion holding one participates in
    per-term logging as ``loss/<stage>/<name>``
    (:class:`~sisr.losses.composite.WeightedSumLoss` is the one that does).
    The tensors are written in place across ordinary steps and only replaced
    when the existing buffer cannot be reused (first use, a device/dtype
    change, or leaving :func:`torch.inference_mode`), so a CUDA-graph replay
    keeps updating the entry it captured.
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
                ``output_range``, ``model_channels``, and ``output_colorspace``.

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
