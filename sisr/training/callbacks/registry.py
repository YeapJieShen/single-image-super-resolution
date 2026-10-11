"""Register a weights file as an MLflow Model: the trained run's, or an official one's."""

import tempfile
from pathlib import Path
from typing import Any

import lightning
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from ... import artifacts
from ..metadata import build_metadata
from ..sinks import _mlflow
from ._common import _logger_step, _sr

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
    """At the end of a fit, register the run's **final** weights as an MLflow Model. Off by default.

    The model registered is the weights as they stand when the fit ends -- not a monitored
    run's best checkpoint. They are written to a temporary ``.safetensors`` file at
    ``on_fit_end`` (so it works whether or not an ``SRWeightsCheckpoint`` kept the final
    file, and for a monitored top-k run), logged to the attached ``MLFlowLogger``'s run with
    their header (``model/metadata.json``), and registered as
    ``sisr-<variant_tag><name_suffix>`` with the tag ``origin=trained``. A run never registers
    silently: with the flag on, a missing ``MLFlowLogger`` is an error.

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
        """Write the final weights and register them when the flag is on.

        Args:
            trainer: The finished trainer.
            pl_module: The trained module; its ``model.variant_tag`` names the model.
        """
        if not self.register_model:
            return
        module = _sr(pl_module)
        tag = module.model.variant_tag
        step = _logger_step(trainer)
        meta = build_metadata(
            module, global_step=trainer.global_step, batch_step=step, epoch=trainer.current_epoch
        )
        logger = self._logger(trainer)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"sisr-{tag}_s{step}{artifacts.SUFFIX}"
            artifacts.save(path, module.model.state_dict(), meta)
            _register(
                logger.experiment,
                logger.run_id,
                path,
                f"sisr-{tag}{self.name_suffix}",
                "trained",
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
