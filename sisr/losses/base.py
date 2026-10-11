"""Abstract base for losses that must adapt to the model's output space."""

import abc

import torch

from ..processors import SRProcessor


class SRLoss(torch.nn.Module, abc.ABC):
    """A criterion that needs facts only the :class:`SRProcessor` knows.

    ``SRLightning`` calls :meth:`bind` once at construction with the processor
    wired alongside the model. Plain ``torch.nn.Module`` criteria (``MSELoss``,
    ``CharbonnierLoss``) need no adaptation and are used directly; this base is
    only for the losses that do.

    **Optional structural protocol:** a loss exposing
    ``last_terms: dict[str, torch.Tensor]`` gets per-term logging as
    ``loss/<stage>/<name>``. ``SRLightning`` checks it with ``getattr``, not
    ``isinstance``, so any criterion holding one participates. Those tensors are
    written **in place** and replaced only when the buffer cannot be reused
    (first use, device/dtype change, leaving :func:`torch.inference_mode`).

    **Second optional structural protocol:** a loss exposing
    ``wants_model_output = True`` (a class or instance attribute) receives
    the model's *full* forward return as its ``pred`` argument -- a bare
    tensor, or an :class:`~sisr.models.base.SRModelOutput` when the model
    returns one -- instead of only the primary tensor every other criterion
    sees. Checked with ``getattr(criterion, "wants_model_output", False)`` in
    ``SRLightning._step``, the same nominal, non-``isinstance`` opt-in shape
    as ``last_terms`` above. A loss that doesn't set it keeps getting
    exactly today's bare primary tensor, whichever shape the model returned.
    """

    @abc.abstractmethod
    def bind(self, processor: SRProcessor) -> None:
        """Adopt the model's output space, or raise if it cannot be served.

        Abstract rather than a defaulted no-op: a wrong inherited default is
        the failure this exists to prevent, and this is the one place an
        impossible model/loss pairing fails loudly rather than silently
        computing the wrong quantity.

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

        Defaulted, not abstract -- presentation must not be a subclass burden.
        """
        return type(self).__name__
