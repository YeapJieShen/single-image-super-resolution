"""Weighted-sum composition of several losses under one criterion."""

import torch

from ..processors import SRProcessor
from .base import SRLoss


class WeightedSumLoss(SRLoss):
    """Sum of named, individually weighted loss terms.

    Every post-SRGAN content loss is a weighted sum, including the one VGG
    recipe reproducible without a discriminator — Ledig et al.'s
    SRResNet-VGG22, which is a VGG22 term plus total variation at ``2e-8``
    and no pixel term at all.

    Terms are **named**, and the names become TensorBoard tag segments
    (``loss/train/{name}``, ``loss/val/{name}``) via :attr:`last_terms`, so
    a run shows which term dominates rather than one opaque total.

    A multi-layer perceptual loss is two
    :class:`~sisr.losses.vgg.VGG19FeatureLoss` terms at different layers —
    there is deliberately no second construction path inside the VGG loss.

    Args:
        terms: Named loss modules, each called as ``term(pred, target)``.
            Plain :class:`torch.nn.Module` losses and
            :class:`~sisr.losses.base.SRLoss` ones may be mixed freely.
        weights: Per-name multipliers. Names absent here default to ``1.0``;
            a name absent from ``terms`` is an error rather than a no-op,
            because the silent alternative is training at the wrong weight.
            Defaults to ``None``.

    Raises:
        ValueError: If ``terms`` is empty, a term name contains ``'/'`` or
            ``'.'``, or ``weights`` names a term that does not exist.
    """

    def __init__(
        self,
        terms: dict[str, torch.nn.Module],
        weights: dict[str, float] | None = None,
    ):
        super().__init__()
        if not terms:
            raise ValueError("WeightedSumLoss needs at least one term")
        for name in terms:
            if "/" in name or "." in name:
                raise ValueError(
                    f"term name {name!r} may not contain '/' or '.' — names become "
                    f"metric-tag segments under loss/<stage>/"
                )
        weights = dict(weights or {})
        unknown = sorted(set(weights) - set(terms))
        if unknown:
            raise ValueError(f"weights names no such term: {unknown}; terms are {sorted(terms)}")

        self.terms = torch.nn.ModuleDict(terms)
        self.weights = {name: float(weights.get(name, 1.0)) for name in terms}
        #: Per-term weighted contributions from the most recent forward, as
        #: stable buffers written in place (see :meth:`forward`) rather than
        #: rebound each call.
        self.last_terms: dict[str, torch.Tensor] = {}

    def bind(self, processor: SRProcessor) -> None:
        """Forward the bind to every term that is an :class:`SRLoss`."""
        for term in self.terms.values():
            if isinstance(term, SRLoss):
                term.bind(processor)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted sum of every term, recording each contribution."""
        total = None
        for name, term in self.terms.items():
            contribution = term(pred, target) * self.weights[name]
            value = contribution.detach()
            prior = self.last_terms.get(name)
            # A buffer created under torch.inference_mode() (trainer.validate()
            # /test()) can never be written to again once outside it, so it must
            # be replaced rather than reused. Otherwise write in place: a
            # replaying backend re-runs only recorded kernels, never the Python
            # line that binds a name, so rebinding would strand the tag on a
            # stale value while loss/train kept moving.
            stale_inference = (
                prior is not None and prior.is_inference() and not torch.is_inference_mode_enabled()
            )
            if (
                prior is None
                or stale_inference
                or prior.device != value.device
                or prior.dtype != value.dtype
            ):
                self.last_terms[name] = value.clone()
            else:
                prior.copy_(value)
            total = contribution if total is None else total + contribution
        assert total is not None  # __init__ requires terms to be non-empty
        return total

    def describe(self) -> str:
        """e.g. ``"1*vgg22 + 2e-08*tv"``."""
        return " + ".join(f"{self.weights[name]:g}*{name}" for name in self.terms)
