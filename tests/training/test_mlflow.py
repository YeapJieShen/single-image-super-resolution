"""MLflow as a second sink: lazy import, scalar fan-out, image artifacts, model registry."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

mlflow = pytest.importorskip("mlflow")  # CI installs `.[dev]` only; the extra is opt-in

import lightning  # noqa: E402
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger  # noqa: E402

from sisr.metrics.scoring import SRScorer  # noqa: E402
from sisr.training import SREvalConfig  # noqa: E402
from sisr.training.callbacks import BenchmarkImageLogger  # noqa: E402
from sisr.training.sinks import MLflowSink, TensorBoardSink, sinks_from  # noqa: E402
from tests.training.test_callbacks import _ignore_gpu_warning, _run_tiny_fit  # noqa: E402


def _db_logger(tmp_path: Path, name: str = "t") -> MLFlowLogger:
    """A local sqlite-backed MLflow (the file store is in maintenance mode in MLflow 3)."""
    return MLFlowLogger(
        experiment_name=name,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=str(tmp_path / "artifacts"),
    )


def _metrics(logger: MLFlowLogger) -> dict[str, list[tuple[int, float]]]:
    client = logger.experiment
    run = client.get_run(logger.run_id)
    return {
        key: [(m.step, m.value) for m in client.get_metric_history(logger.run_id, key)]
        for key in run.data.metrics
    }


# --- 1. the extra is optional and lazy --------------------------------------------------


def test_importing_the_training_package_does_not_import_mlflow():
    code = "import sys, sisr.training; assert 'mlflow' not in sys.modules, 'mlflow imported'"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_a_missing_extra_is_an_actionable_error_not_a_bare_module_not_found(monkeypatch):
    from sisr.training import sinks

    monkeypatch.setitem(sys.modules, "mlflow", None)  # `import mlflow` now raises ImportError
    with pytest.raises(ImportError, match=r"pip install '\.\[mlflow\]'") as err:
        sinks._mlflow()
    assert not isinstance(err.value, ModuleNotFoundError)


# --- 2. scalar fan-out needs no code in the loop ----------------------------------------


@_ignore_gpu_warning
def test_a_tensorboard_and_an_mlflow_logger_both_receive_the_scalars(tmp_path):

    tb = TensorBoardLogger(str(tmp_path / "tb"))
    ml = _db_logger(tmp_path)
    _run_tiny_fit(callbacks=[], n_batches=4, logger=[tb, ml], log_every_n_steps=1)

    logged = _metrics(ml)
    assert "loss/train" in logged and len(logged["loss/train"]) >= 2
    assert any(tb.log_dir and Path(tb.log_dir).glob("events.out.tfevents.*"))


# --- 3. image strips and per-image scalars reach MLflow on the logger step axis --------


def _bench_cb(tmp_path):
    ml = _db_logger(tmp_path)
    cb = BenchmarkImageLogger(log_per_image_metrics=True)
    cb._buffer["Set5"] = []
    cb._tb_experiment = MagicMock()
    cb._extra_sinks = [MLflowSink(ml)]
    return cb, ml


def _collect(cb, trainer, global_step_axis_gap=True):
    eval_config = SREvalConfig(crop_border=0, psnr_channels=["RGB"], ssim_channels=[])
    pl_module = MagicMock()
    pl_module.eval_config = eval_config
    pl_module.scorer = SRScorer(eval_config)
    pl_module.predict_rgb = MagicMock(return_value=(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)))
    cb._collect_batch(
        trainer,
        pl_module,
        batch=(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)),
        batch_idx=0,
        dataset_name="Set5",
        source_dataloaders=[
            SimpleNamespace(dataset=SimpleNamespace(img_paths=[Path("baby.png")]), batch_size=1)
        ],
        dataloader_idx=0,
        should_log_images=True,
    )


def _trainer(global_step: int, batches: int):
    return SimpleNamespace(
        global_step=global_step,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=batches)),
        loggers=[],
    )


def test_the_strip_lands_on_mlflow_at_the_same_step_tensorboard_gets_it(tmp_path):
    cb, ml = _bench_cb(tmp_path)
    # Two-optimizer fixture: the optimizer counter is 2x the batch-counted logger axis.
    _collect(cb, _trainer(global_step=400, batches=200))

    tb_step = cb._tb_experiment.add_image.call_args.kwargs["global_step"]
    artifacts = [a.path for a in ml.experiment.list_artifacts(ml.run_id, "benchmark/Set5")]
    assert tb_step == 200
    assert artifacts == ["benchmark/Set5/baby_step200.png"]  # not _step400: trap 10
    scalars = _metrics(ml)
    key = "per_image/Set5/psnr/RGB/baby"
    assert scalars[key] == [(200, scalars[key][0][1])]


def test_a_run_with_no_logger_at_all_still_skips_every_write_and_the_step(tmp_path):
    cb = BenchmarkImageLogger(log_per_image_metrics=True)
    cb._buffer["Set5"] = []
    cb._tb_experiment = None
    cb._extra_sinks = []
    _collect(cb, SimpleNamespace(loggers=[]))  # no fit_loop: reading the step would raise
    assert len(cb._buffer["Set5"]) == 1


def test_sinks_from_returns_one_sink_per_supported_logger(tmp_path):
    tb, ml = TensorBoardLogger(str(tmp_path / "tb")), _db_logger(tmp_path)
    csv = lightning.pytorch.loggers.CSVLogger(str(tmp_path / "csv"))

    sinks = sinks_from(SimpleNamespace(loggers=[tb, ml, csv]))
    assert [type(s) for s in sinks] == [TensorBoardSink, MLflowSink]
    assert sinks_from(SimpleNamespace(loggers=[csv])) == []


def test_weight_histograms_stay_tensorboard_only(tmp_path):
    from sisr.training.callbacks import WeightHistogramLogger

    ml = _db_logger(tmp_path)
    cb = WeightHistogramLogger()
    pl_module = MagicMock()
    pl_module.named_parameters.return_value = [("model.w", torch.nn.Parameter(torch.rand(3)))]
    trainer = SimpleNamespace(
        loggers=[ml],
        global_step=1,
        current_epoch=0,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=0)),
    )
    cb.on_train_batch_end(trainer, pl_module, None, None, 0)  # TB-only: no MLflow write
    assert _metrics(ml) == {}


def test_setup_attaches_an_mlflow_sink_beside_the_tensorboard_experiment(tmp_path):
    cb = BenchmarkImageLogger(dataset_names=["Set5"])
    tb, ml = TensorBoardLogger(str(tmp_path / "tb")), _db_logger(tmp_path)
    cb.setup(SimpleNamespace(datamodule=None, loggers=[tb, ml]), pl_module=None, stage="fit")

    assert cb._tb_experiment is tb.experiment
    assert [type(s) for s in cb._extra_sinks] == [MLflowSink]
    assert [type(s) for s in cb._sinks()] == [TensorBoardSink, MLflowSink]
    only_ml = BenchmarkImageLogger(dataset_names=["Set5"])
    only_ml.setup(SimpleNamespace(datamodule=None, loggers=[ml]), pl_module=None, stage="fit")
    assert only_ml._tb_experiment is None and [type(s) for s in only_ml._sinks()] == [MLflowSink]


# --- 4. the model registry: opt-in, and trained vs official are different names ----------


def _fit_with_registrar(tmp_path, **registrar_kw):
    from sisr.training.callbacks import MLflowModelRegistrar, SRWeightsCheckpoint

    ml = _db_logger(tmp_path)
    weights = SRWeightsCheckpoint(
        monitor_metric=None,
        keep_last=2,
        every_n_train_steps=2,
        dirpath=str(tmp_path / "w"),
        filename_prefix="sr-weights",
    )
    _run_tiny_fit(
        callbacks=[weights, MLflowModelRegistrar(**registrar_kw)],
        n_batches=4,
        logger=ml,
        log_every_n_steps=1,
    )
    return ml


def _registered(ml):
    return sorted(m.name for m in ml.experiment.search_registered_models())


@_ignore_gpu_warning
def test_the_registry_flag_defaults_off(tmp_path):
    assert _registered(_fit_with_registrar(tmp_path)) == []


@_ignore_gpu_warning
def test_the_flag_on_registers_the_final_weights_under_the_variant_tag(tmp_path):
    ml = _fit_with_registrar(tmp_path, register_model=True)

    assert _registered(ml) == ["sisr-313"]
    (version,) = ml.experiment.search_model_versions("name='sisr-313'")
    files = [a.path for a in ml.experiment.list_artifacts(ml.run_id, "model")]
    assert sorted(files) == ["model/metadata.json", "model/sr-weights_s3.safetensors"]
    assert version.run_id == ml.run_id and version.tags["origin"] == "trained"


@_ignore_gpu_warning
def test_a_name_suffix_extends_the_trained_name(tmp_path):
    assert _registered(_fit_with_registrar(tmp_path, register_model=True, name_suffix="-x")) == [
        "sisr-313-x"
    ]


def test_the_flag_on_without_an_mlflow_logger_is_an_actionable_error_at_setup(tmp_path):
    from lightning.pytorch.utilities.exceptions import MisconfigurationException

    from sisr.training.callbacks import MLflowModelRegistrar

    cb = MLflowModelRegistrar(register_model=True)
    with pytest.raises(MisconfigurationException, match="MLFlowLogger"):
        cb.setup(SimpleNamespace(loggers=[]), pl_module=None, stage="fit")


def test_official_weights_register_under_a_distinct_official_name(tmp_path):
    from sisr import artifacts
    from sisr.training.callbacks import register_official_weights

    path = tmp_path / "official.safetensors"
    artifacts.save(path, {"w": torch.zeros(2)}, {"model": {"class_path": "x"}})
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    name = register_official_weights(
        path, "80B64F", tracking_uri=uri, artifact_location=str(tmp_path / "art")
    )

    assert name == "sisr-80B64F-official"
    client = mlflow.MlflowClient(tracking_uri=uri)
    (version,) = client.search_model_versions("name='sisr-80B64F-official'")
    assert version.tags["origin"] == "official"
    assert name != "sisr-80B64F"


# --- 5. no template turns tracking on by itself ------------------------------------------


def test_no_template_sets_a_tracking_uri_or_an_mlflow_logger_uncommented():
    import yaml

    def walk(o, name):
        if isinstance(o, dict):
            assert "tracking_uri" not in o, name
            assert "MLFlowLogger" not in str(o.get("class_path", "")), name
            for v in o.values():
                walk(v, name)
        elif isinstance(o, list):
            for v in o:
                walk(v, name)

    templates = Path(__file__).resolve().parents[2] / "templates"
    for path in sorted(templates.glob("*.yaml")):
        walk(yaml.safe_load(path.read_text(encoding="ascii")), path.name)
    shown = [p.name for p in templates.glob("*.yaml") if "MLFlowLogger" in p.read_text("ascii")]
    assert shown == ["config.srcnn.template.yaml", "config.srresnet.template.yaml"]


# --- 6. a real tracking server, when one answers ----------------------------------------


def _server_uri() -> str | None:
    import os
    import urllib.request

    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    if not uri.startswith("http"):
        return None
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=2) as resp:
            return uri if resp.status == 200 else None
    except OSError:
        return None


@pytest.mark.skipif(_server_uri() is None, reason="no MLflow tracking server answering /health")
@_ignore_gpu_warning
def test_a_live_server_receives_scalars_and_the_strip_in_its_own_experiment():
    uri = _server_uri()
    ml = MLFlowLogger(experiment_name="sisr-smoke", tracking_uri=uri)  # never a real experiment
    _run_tiny_fit(callbacks=[], n_batches=4, logger=ml, log_every_n_steps=1)

    cb = BenchmarkImageLogger(log_per_image_metrics=True)
    cb._buffer["Set5"] = []
    cb._extra_sinks = [MLflowSink(ml)]
    _collect(cb, _trainer(global_step=6, batches=3))

    assert "loss/train" in _metrics(ml)
    assert [a.path for a in ml.experiment.list_artifacts(ml.run_id, "benchmark/Set5")] == [
        "benchmark/Set5/baby_step3.png"
    ]
    ml.experiment.set_terminated(ml.run_id)
