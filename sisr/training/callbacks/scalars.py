"""Diagnostic scalar callbacks: gradient norm, weight histograms, loss-weight ramps."""

from typing import Any

import lightning
import torch
import torch.nn.functional
from lightning.pytorch.callbacks import (
    Callback,
)

from ._common import _logger_step


class GradNormLogger(Callback):
    """Log the total gradient L2 norm to TensorBoard periodically.

    Computes ``sqrt(sum(p.grad.norm() ** 2))`` across all model parameters
    after the backward pass and logs it as the ``"diag/grad_norm"`` scalar.

    Args:
        every_n_batches (int): Compute and log every *n* batches. Defaults to
            ``100``. The unit is in the name deliberately: Lightning's own
            callbacks do the same (``every_n_train_steps``, ``every_n_epochs``),
            and the old name was a homonym of ``Trainer.log_every_n_steps``,
            which means something else — the metric *flush* cadence. Batches
            also make the setting mean one thing under both automatic and manual
            optimization, and match the axis the emitted scalar is plotted on.
            See :func:`_logger_step`.
    """

    def __init__(self, every_n_batches: int = 100):
        super().__init__()
        self.every_n_batches = every_n_batches

    def on_after_backward(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Compute and log gradient norm if on the right step cadence.

        Per-parameter norms are stacked and reduced with a single
        ``torch.linalg.vector_norm`` call so only one ``.item()`` forces a
        GPU sync, instead of one per parameter (the previous per-parameter
        ``.item()`` accumulation loop).

        Args:
            trainer: The trainer instance.
            pl_module: The model being trained.
        """
        if _logger_step(trainer) % self.every_n_batches != 0:
            return

        grad_norms = [p.grad.detach().norm(2) for p in pl_module.parameters() if p.grad is not None]
        total_norm = (
            torch.linalg.vector_norm(torch.stack(grad_norms), ord=2).item() if grad_norms else 0.0
        )

        pl_module.log("diag/grad_norm", total_norm, on_step=True, on_epoch=False, sync_dist=True)


class LossWeightScheduler(Callback):
    """Linearly ramps one ``WeightedSumLoss`` term's weight over training.

    Generic and YAML-configurable, so an architecture whose paper ramps a
    loss term over training (e.g. DRCN's recursive-supervision weight)
    needs no bespoke schedule mechanism of its own -- it points this at the
    term's name and a start/end/span.

    Reads ``pl_module.criterion`` structurally (``getattr(..., None)`` plus
    ``callable``, the same structural opt-in idiom
    :class:`~sisr.losses.base.SRLoss` already establishes for ``last_terms``)
    rather than by ``isinstance``, so any criterion exposing that method
    participates, not only :class:`~sisr.losses.WeightedSumLoss` by name.

    Progress is measured on the batch-counted step axis (:func:`_logger_step`,
    the same axis every ``self.log`` metric uses), not ``trainer.global_step``
    -- SRGAN's manual optimization steps two optimizers per batch, which would
    otherwise make the configured span mean half as many batches under
    adversarial training as under automatic optimization.

    Args:
        term: Name of the term inside the criterion's weights to schedule.
        start_value: Weight at or before ``start_step``.
        end_value: Weight at or after ``start_step + duration``.
        duration: Number of steps the ramp spans. Must be positive.
        start_step: Step at which the ramp begins. Defaults to ``0``.

    Raises:
        ValueError: If ``duration`` is not positive.
    """

    def __init__(
        self,
        term: str,
        start_value: float,
        end_value: float,
        duration: int,
        start_step: int = 0,
    ):
        super().__init__()
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}")
        self.term = term
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration
        self.start_step = start_step

    def _value_at(self, step: int) -> float:
        """The scheduled weight at a given (batch-counted) step."""
        if step <= self.start_step:
            return self.start_value
        if step >= self.start_step + self.duration:
            return self.end_value
        frac = (step - self.start_step) / self.duration
        return self.start_value + (self.end_value - self.start_value) * frac

    def on_train_batch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Set the term's weight for the batch about to run.

        Args:
            trainer: The trainer instance.
            pl_module: The model being trained -- ``pl_module.criterion``
                must expose ``set_weight(name, value)``.
            batch: Unused.
            batch_idx: Unused -- progress reads the trainer's own step axis.

        Raises:
            TypeError: If ``pl_module.criterion`` has no ``set_weight``
                method, e.g. a plain :class:`torch.nn.MSELoss` rather than a
                :class:`~sisr.losses.WeightedSumLoss`.
        """
        set_weight = getattr(pl_module.criterion, "set_weight", None)
        if not callable(set_weight):
            raise TypeError(
                f"LossWeightScheduler needs pl_module.criterion to expose "
                f"set_weight(name, value) (e.g. WeightedSumLoss); got "
                f"{type(pl_module.criterion).__name__}"
            )
        set_weight(self.term, self._value_at(_logger_step(trainer)))


class WeightHistogramLogger(Callback):
    """Logs model weights as TensorBoard histograms, grouped by parameter prefix.

    Groups by prefix, e.g. ``model.feat``, ``model.mapping``, ``model.recon``.

    Args:
        every_n_batches (int): Log histograms every *n* batches, matching the
            axis they are written on — see :func:`_logger_step`. Named for its
            unit, following Lightning's own ``every_n_*`` callback arguments.
            Defaults to ``10000``. Histograms dominate event-file size, and
            one is written per tracked parameter on every cadence hit — at
            the templates' 1M-step schedule the old default of ``100`` was
            ~10k writes per parameter (multi-GB event files) despite never
            having been exercised at that scale (neither tracked template
            wires this callback). ``10000`` drops that to ~100
            writes per parameter, still enough to see the weight-drift
            trend across a full run.
    """

    def __init__(self, every_n_batches: int = 10000):
        super().__init__()
        self.every_n_batches = every_n_batches

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log grouped weights as histograms if on the right step cadence.

        Args:
            trainer: The trainer instance.
            pl_module: The model being trained.
            outputs: Unused.
            batch: Unused.
            batch_idx: Unused.
        """
        step = _logger_step(trainer)
        if step % self.every_n_batches != 0:
            return

        tb_logger = next(
            (
                logger
                for logger in trainer.loggers
                if isinstance(logger, lightning.pytorch.loggers.TensorBoardLogger)
            ),
            None,
        )
        if tb_logger is None:
            return

        experiment = tb_logger.experiment

        for name, param in pl_module.named_parameters():
            if param.requires_grad and name.startswith("model."):
                parts = name.split(".", 2)
                tb_name = parts[0] + "." + "/".join(parts[1:])
                experiment.add_histogram(tb_name, param, global_step=step)
