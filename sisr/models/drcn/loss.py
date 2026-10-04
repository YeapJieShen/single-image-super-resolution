"""Recursive supervision for DRCN."""

import torch

from sisr.losses.base import SRLoss
from sisr.models.base import SRModelOutput
from sisr.processors import SRProcessor


class RecursiveSupervisionLoss(SRLoss):
    """Mean squared error of every recursion's prediction against the target.

    This is DRCN's ``l1`` term (Kim et al. 2016), ``1/D * sum_d ||y - y_d||^2``.
    Combine it with the ensemble's own MSE under a
    :class:`~sisr.losses.composite.WeightedSumLoss` and two
    :class:`~sisr.training.callbacks.LossWeightScheduler` instances to get
    ``alpha * l1 + (1 - alpha) * l2``.

    Opts in to the model's full output record (``wants_model_output``) and
    reads ``extras["recursions"]``, shape ``(B, D, C, H, W)``.
    """

    wants_model_output = True

    def bind(self, processor: SRProcessor) -> None:
        """No-op: the loss is a plain MSE in the model's own space."""

    def forward(self, pred: SRModelOutput, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Mean over recursions of the per-recursion MSE.

        Args:
            pred: The model's output record, carrying ``extras["recursions"]``.
            target: ``(B, C, H, W)`` target in model space.

        Raises:
            TypeError: If ``pred`` is a bare tensor rather than a record.
        """
        if not isinstance(pred, SRModelOutput) or "recursions" not in pred.extras:
            raise TypeError(
                "RecursiveSupervisionLoss needs the model's SRModelOutput with "
                "extras['recursions'] (a recursive model such as DRCN); got "
                f"{type(pred).__name__}."
            )
        recursions = pred.extras["recursions"]
        return ((recursions - target.unsqueeze(1)) ** 2).mean()
