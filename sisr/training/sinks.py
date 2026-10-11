"""Where callbacks that write straight to a logger's client send images and scalars.

Lightning fans ``self.log`` scalars out to every attached logger by itself; what it cannot
do is the writes a callback makes to one logger's *client* (an image strip, a per-image
scalar). :class:`MetricSink` is that seam: :func:`sinks_from` returns one sink per attached
logger that has one, so a callback writes through the list instead of branching on logger
types. Callers stamp ``step`` with :func:`~sisr.training.callbacks._common._logger_step`
and pass it in unchanged -- a sink never reads a step counter itself (trap 10).

``mlflow`` is imported inside :func:`_mlflow` only, so nothing here, nor anything that
imports :mod:`sisr.training`, pays for it or needs the ``[mlflow]`` extra unless MLflow is
actually used.
"""

import re
from typing import TYPE_CHECKING, Any, Protocol

import lightning
import torch

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger

__all__ = ["MLflowSink", "MetricSink", "TensorBoardSink", "sinks_from"]


def _mlflow() -> Any:
    """Import ``mlflow`` on demand, turning a missing extra into an actionable error."""
    try:
        import mlflow
    except ImportError as err:
        raise ImportError(
            "MLflow tracking needs the `mlflow` package. Install the extra: "
            "pip install '.[mlflow]'."
        ) from err
    return mlflow


class MetricSink(Protocol):
    """One destination for the images and scalars a callback writes directly."""

    def image(self, tag: str, chw: torch.Tensor, step: int) -> None:
        """Write a ``(3, H, W)`` float image in ``[0, 1]`` under ``tag`` at ``step``."""

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Write one scalar under ``tag`` at ``step``."""


class TensorBoardSink:
    """A TensorBoard ``SummaryWriter``.

    Args:
        experiment: The logger's ``experiment`` (``torch.utils.tensorboard.SummaryWriter``).
    """

    def __init__(self, experiment: Any) -> None:
        self.experiment = experiment

    def image(self, tag: str, chw: torch.Tensor, step: int) -> None:
        """See :meth:`MetricSink.image`."""
        self.experiment.add_image(tag, chw, global_step=step)

    def scalar(self, tag: str, value: float, step: int) -> None:
        """See :meth:`MetricSink.scalar`."""
        self.experiment.add_scalar(tag, value, global_step=step)


_KEY_UNSAFE = re.compile(r"[^0-9A-Za-z_\-. /]")


class MLflowSink:
    """An MLflow run, through the ``MlflowClient`` of an attached ``MLFlowLogger``.

    Images are run artifacts at ``benchmark/{tag}_step{step}.png``; scalars are metrics at
    ``step``. MLflow restricts metric names to ``[0-9A-Za-z_-. /]``, so any other character
    of a tag (a benchmark image's file stem, say) becomes ``_``.

    Args:
        logger: The attached ``MLFlowLogger``.
    """

    def __init__(self, logger: "MLFlowLogger") -> None:
        self._mlflow = _mlflow()  # fail here, at setup, rather than mid-evaluation
        self.client = logger.experiment
        self.run_id = logger.run_id

    def image(self, tag: str, chw: torch.Tensor, step: int) -> None:
        """See :meth:`MetricSink.image`."""
        hwc = (chw.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        path = f"benchmark/{_KEY_UNSAFE.sub('_', tag)}_step{step}.png"
        self.client.log_image(self.run_id, hwc.numpy(), artifact_file=path)

    def scalar(self, tag: str, value: float, step: int) -> None:
        """See :meth:`MetricSink.scalar`."""
        self.client.log_metric(self.run_id, _KEY_UNSAFE.sub("_", tag), float(value), step=step)


def sinks_from(trainer: lightning.Trainer) -> list[MetricSink]:
    """One sink per attached logger that has one (TensorBoard, MLflow); none if neither.

    Args:
        trainer: The trainer whose ``loggers`` to read.

    Returns:
        The sinks, in logger order. Empty when no supported logger is attached.
    """
    from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

    sinks: list[MetricSink] = []
    for logger in getattr(trainer, "loggers", None) or []:
        if isinstance(logger, TensorBoardLogger):
            sinks.append(TensorBoardSink(logger.experiment))
        elif isinstance(logger, MLFlowLogger):
            sinks.append(MLflowSink(logger))
    return sinks
