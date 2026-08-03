"""End-to-end Trainer integration for the SR training stack.

A single ``Trainer(fast_dev_run=True)`` fit+test over the tiny fixture images
exercises the wiring the per-method unit tests in ``test_lightning_module.py``
cannot reach: ``training_step`` / ``validation_step`` under a real loop,
``dataloader_idx`` routing across the ``[primary_val, Set5]`` val-loader list,
the metric names actually logged, and the module -> datamodule ->
``BenchmarkImageLogger`` integration (the callback auto-discovers ``Set5`` from
``datamodule.test_names``). It is also the suite's canary for the lightning
``LeafSpec`` pytree ``FutureWarning``: only a real Trainer loop emits it,
and the strict ``filterwarnings=error`` config turns it into a failure unless
the scoped ignore in ``pyproject.toml`` is in place.
"""

import functools
from pathlib import Path

import lightning
import pytest
import torch

from sisr.models.srcnn import SRCNN
from sisr.models.srresnet import SRResNetTrainingConfig
from sisr.models.srresnet.model import SRResNet
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import (
    BenchmarkImageLogger,
    SRDataModule,
    SREvalConfig,
    SRLightning,
    SRPredictionWriter,
)


def _make_srcnn_module() -> SRLightning:
    """SRLightning wrapping a 3-channel SRCNN with the RGB pass-through processor."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def _make_datamodule(image_dir: Path) -> SRDataModule:
    """SRDataModule over one tiny image dir: train + primary val + one Set5 test set."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "use_tqdm": False,
            "cache_dir": str(image_dir / ".lmdb_cache_integration"),
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(image_dir), "scale": 2},
    }
    return SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        test_datasets={"Set5": val_spec},
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
        test_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


# num_workers=0 (tiny fixture) and accelerator="cpu" on a CUDA box both raise
# PossibleUserWarnings — environment noise, not a library-health signal — so
# silence them for this test only, leaving the strict global filter honest elsewhere.
@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_fast_dev_run_fit_and_test_logs_module_and_callback_metrics(
    tiny_rgb_image_dir: Path,
):
    """Fails if a real fit+test loop stops logging the expected metric keys.

    Catches wiring regressions the unit tests can't: a broken ``training_step`` /
    ``validation_step``, mis-routed ``dataloader_idx``, renamed metric keys, or a
    ``BenchmarkImageLogger`` that no longer discovers the datamodule's test set.
    ``accelerator="cpu"`` keeps CI (no GPU) and local (GPU) runs identical.
    """
    module = _make_srcnn_module()
    datamodule = _make_datamodule(tiny_rgb_image_dir)
    trainer = lightning.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[BenchmarkImageLogger(log_every_n_val_runs=1)],
    )

    trainer.fit(module, datamodule=datamodule)
    fit_metrics = set(trainer.callback_metrics)
    # Module-logged: training_step + validation_step on the primary val loader (idx 0).
    # ssim/val/{RGB,Y} — eval_config.ssim_channels defaults to ['RGB', 'Y'].
    assert {"loss/train", "loss/val", "psnr/val/RGB", "ssim/val/RGB", "ssim/val/Y"} <= fit_metrics
    # Callback-logged for the Set5 test loader (val_dataloader idx 1) — proves
    # BenchmarkImageLogger auto-discovered the datamodule's test set and ran.
    assert {"psnr/Set5/RGB", "ssim/Set5/RGB", "ssim/Set5/Y"} <= fit_metrics

    trainer.test(module, datamodule=datamodule)
    test_metrics = set(trainer.callback_metrics)
    # test_step is a no-op; these come solely from BenchmarkImageLogger.on_test_*.
    assert {"psnr/Set5/RGB", "ssim/Set5/RGB", "ssim/Set5/Y"} <= test_metrics


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_fast_dev_run_raises_on_cross_wired_model_and_dataset_through_real_trainer(
    tiny_rgb_image_dir: Path,
):
    """Regression: SRLightning.setup()'s input_contract probe must actually
    fire through a real Trainer.fit() call, not only the SimpleNamespace-
    stubbed trainer every setup() unit test in test_lightning_module.py uses.
    Locks Lightning's documented DataModule.setup-before-LightningModule.setup
    hook ordering (Trainer._call_setup_hook) against a silent regression that
    every stubbed-trainer test would stay green through.
    """
    model = SRResNet(scale=2, num_residual_blocks=1)
    module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRResNetTrainingConfig(scale=2),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    # srcnn's datasets are pre_upsampled (lr.shape == hr.shape) -- mismatched
    # against SRResNet's native_lr contract.
    datamodule = _make_datamodule(tiny_rgb_image_dir)
    trainer = lightning.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with pytest.raises(ValueError, match="input_contract"):
        trainer.fit(module, datamodule=datamodule)


# ---------------------------------------------------------------------------
# cli predict end-to-end — LR-only inference path, both architectures
# ---------------------------------------------------------------------------


def _make_predict_datamodule(image_dir: Path, arch: str) -> SRDataModule:
    """Predict-only SRDataModule: train_dataset/val_dataset are required by
    the constructor but never instantiated — setup(stage='predict') only
    builds predict_dataset, matching every other stage's selective build."""
    unused_spec = {
        "class_path": f"sisr.datasets.{arch}.ValidationDataset",
        "init_args": {"img_dir": str(image_dir), "scale": 2},
    }
    return SRDataModule(
        train_dataset=unused_spec,
        val_dataset=unused_spec,
        predict_dataset={
            "class_path": "sisr.datasets.predict.PredictDataset",
            "init_args": {"img_dir": str(image_dir)},
        },
        predict_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_predict_end_to_end_srcnn_y_channel_same_size_output(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """SRCNN's pre-upsampled, Y-channel path: predict output must be the same
    H/W as the input (scale=1x) and land as one PNG per input file."""
    from PIL import Image

    model = SRCNN(num_channels=1, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    module = SRLightning(
        model=model,
        processor=YChannelProcessor(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    datamodule = _make_predict_datamodule(tiny_rgb_image_dir, arch="srcnn")
    out_dir = tmp_path / "predictions"
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[SRPredictionWriter(output_dir=out_dir)],
    )

    predictions = trainer.predict(module, datamodule=datamodule)

    assert len(predictions) == 3  # tiny_rgb_image_dir has 3 images, batch_size=1
    for pred in predictions:
        assert pred.shape == (1, 3, 36, 36)  # 36x36 input, scale=1x

    written = sorted(p.stem for p in out_dir.glob("*.png"))
    assert written == ["img_00", "img_01", "img_02"]
    for name in written:
        with Image.open(out_dir / f"{name}.png") as img:
            assert img.size == (36, 36)


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_predict_end_to_end_srresnet_rgb_upsamples_by_scale(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """SRResNet's genuine-LR, RGB path: predict output must be exactly
    scale x the input H/W."""
    from PIL import Image

    model = SRResNet(scale=2, num_residual_blocks=1)
    module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    datamodule = _make_predict_datamodule(tiny_rgb_image_dir, arch="srresnet")
    out_dir = tmp_path / "predictions"
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[SRPredictionWriter(output_dir=out_dir)],
    )

    predictions = trainer.predict(module, datamodule=datamodule)

    assert len(predictions) == 3
    for pred in predictions:
        assert pred.shape == (1, 3, 72, 72)  # 36x36 input, scale=2x -> 72x72

    written = sorted(p.stem for p in out_dir.glob("*.png"))
    assert written == ["img_00", "img_01", "img_02"]
    for name in written:
        with Image.open(out_dir / f"{name}.png") as img:
            assert img.size == (72, 72)
