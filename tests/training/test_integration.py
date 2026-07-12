"""End-to-end Trainer integration for the SR training stack (P3.1).

A single ``Trainer(fast_dev_run=True)`` fit+test over the tiny fixture images
exercises the wiring the per-method unit tests in ``test_lightning_module.py``
cannot reach: ``training_step`` / ``validation_step`` under a real loop,
``dataloader_idx`` routing across the ``[primary_val, Set5]`` val-loader list,
the metric names actually logged, and the module -> datamodule ->
``BenchmarkImageLogger`` integration (the callback auto-discovers ``Set5`` from
``datamodule.test_names``). It is also the suite's canary for the lightning
``LeafSpec`` pytree ``FutureWarning`` (P4.8): only a real Trainer loop emits it,
and the strict ``filterwarnings=error`` config turns it into a failure unless
the scoped ignore in ``pyproject.toml`` is in place.
"""

import functools
from pathlib import Path

import lightning
import pytest
import torch

from sisr.models.srcnn import SRCNN
from sisr.processors import RGBProcessor
from sisr.training import (
    BenchmarkImageLogger,
    SRDataModule,
    SREvalConfig,
    SRLightning,
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
            "blur_sigma": 1.0,
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
# lightning PossibleUserWarnings — config/environment noise, not a library-health
# signal. Silence them for THIS test only so the strict global filter stays honest
# and the test fails specifically on the real FutureWarning when the filter is stale.
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
    assert {"train_loss", "val_loss", "val_psnr(RGB)", "val_ssim"} <= fit_metrics
    # Callback-logged for the Set5 test loader (val_dataloader idx 1) — proves
    # BenchmarkImageLogger auto-discovered the datamodule's test set and ran.
    assert {"Set5_psnr(RGB)", "Set5_ssim"} <= fit_metrics

    trainer.test(module, datamodule=datamodule)
    test_metrics = set(trainer.callback_metrics)
    # test_step is a no-op; these come solely from BenchmarkImageLogger.on_test_*.
    assert {"Set5_psnr(RGB)", "Set5_ssim"} <= test_metrics
