"""Register a weights file as an MLflow Model: the trained run's, or an official one's."""

from pathlib import Path
from typing import Any, cast

import lightning
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from ... import artifacts
from ..sinks import _mlflow
from .checkpoint import SRWeightsCheckpoint

__all__ = ["MLflowModelRegistrar", "register_official_weights"]


def _register(client: Any, run_id: str, path: Path, name: str, origin: str) -> str:
    """Log ``path`` and its header under ``model/`` of the run, and version it as ``name``."""
    _, meta = artifacts.load(path)
    client.log_artifact(run_id, str(path), artifact_path="model")
    client.log_dict(run_id, meta, "model/metadata.json")
    try:
        client.create_registered_model(name)
    except _mlflow().exceptions.MlflowException:  # already registered: add a version
        pass
    client.create_model_version(
        name, source=f"runs:/{run_id}/model", run_id=run_id, tags={"origin": origin}
    )
    return name


class MLflowModelRegistrar(Callback):
    """At the end of a fit, register the run's weights as an MLflow Model. Off by default.

    The file is the first ``SRWeightsCheckpoint`` over the generator's own ``model``: its
    best file when it monitors a metric, else its newest rolling file. It is logged to the
    attached ``MLFlowLogger``'s run together with its header (``model/metadata.json``) and
    registered as ``sisr-<variant_tag><name_suffix>`` with the tag ``origin=trained``. A run
    never registers silently: with the flag on, a missing ``MLFlowLogger`` or weights
    callback is an error.

    Args:
        register_model: Register at ``on_fit_end``. Default ``False``.
        name_suffix: Appended to the registered name, e.g. ``"-ablation"``.
    """

    def __init__(self, register_model: bool = False, name_suffix: str = ""):
        super().__init__()
        self.register_model = register_model
        self.name_suffix = name_suffix

    @staticmethod
    def _logger(trainer: lightning.Trainer) -> Any:
        from lightning.pytorch.loggers import MLFlowLogger

        return next((k for k in trainer.loggers if isinstance(k, MLFlowLogger)), None)

    def setup(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule, stage: str
    ) -> None:
        """Fail early, with the fix, if registration is on but cannot work.

        Args:
            trainer: The active trainer.
            pl_module: Unused.
            stage: Unused.

        Raises:
            MisconfigurationException: If ``register_model`` is on and no ``MLFlowLogger``
                is attached.
        """
        if self.register_model and self._logger(trainer) is None:
            raise MisconfigurationException(
                "MLflowModelRegistrar(register_model=True) needs a "
                "lightning.pytorch.loggers.MLFlowLogger in trainer.logger; none is attached."
            )

    def on_fit_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        """Register the final weights when the flag is on.

        Args:
            trainer: The finished trainer.
            pl_module: The trained module; its ``model.variant_tag`` names the model.

        Raises:
            MisconfigurationException: If no ``SRWeightsCheckpoint`` over ``model`` holds a file.
        """
        if not self.register_model:
            return
        ckpt = next(
            (
                c
                for c in cast(Any, trainer).callbacks
                if isinstance(c, SRWeightsCheckpoint) and c.attribute == "model"
            ),
            None,
        )
        path = (
            (ckpt.best_model_path or (ckpt._rolling[-1] if ckpt._rolling else "")) if ckpt else ""
        )
        if not path:
            raise MisconfigurationException(
                "register_model=True found no saved weights: add an SRWeightsCheckpoint "
                "(attribute 'model') that has written a file by the end of the fit."
            )
        logger = self._logger(trainer)
        tag = pl_module.model.variant_tag  # type: ignore[union-attr]
        _register(
            logger.experiment, logger.run_id, Path(path), f"sisr-{tag}{self.name_suffix}", "trained"
        )


def register_official_weights(
    path: str | Path,
    variant_tag: str,
    tracking_uri: str,
    artifact_location: str | None = None,
) -> str:
    """Register an externally sourced, already parity-scored weights file as the official model.

    The name is ``sisr-<variant_tag>-official``, so it can never collide with a locally
    trained run's ``sisr-<variant_tag>``; the version carries ``origin=official``. The file is
    logged to a run in the ``sisr-official`` experiment.

    Args:
        path: A ``.safetensors`` file in this project's format (header included).
        variant_tag: The model's ``variant_tag``, e.g. ``'80B64F'``.
        tracking_uri: The MLflow tracking URI to register in. Required: nothing is implied.
        artifact_location: Artifact root for the ``sisr-official`` experiment if this call
            creates it; ``None`` leaves it to the server's default.

    Returns:
        The registered model name.

    Raises:
        ImportError: If the ``mlflow`` extra is not installed.
    """
    client = _mlflow().MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("sisr-official")
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment("sisr-official", artifact_location)
    )
    run = client.create_run(experiment_id, run_name=f"register-{variant_tag}")
    try:
        return _register(
            client, run.info.run_id, Path(path), f"sisr-{variant_tag}-official", "official"
        )
    finally:
        client.set_terminated(run.info.run_id)
