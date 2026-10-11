"""VDSR through the real Lightning module: multi-scale data, per-scale tags, LR-scaled clipping."""

from pathlib import Path

import lightning
import pytest
import torch

from sisr.models.vdsr import VDSR, VDSREvalConfig, VDSRTrainingConfig
from sisr.processors import YChannelProcessor
from sisr.training import BenchmarkImageLogger, SRDataModule, SRLightning


def _lit(**train_kwargs) -> SRLightning:
    torch.manual_seed(0)
    return SRLightning(
        model=VDSR(depth=3, features=4),
        processor=YChannelProcessor(),
        training_config=VDSRTrainingConfig(scale=[2, 3], **train_kwargs),
        eval_config=VDSREvalConfig(ssim_channels=[]),
        optimizer=lambda params: torch.optim.SGD(params, lr=0.5, momentum=0.9),
    )


def _dm(path: Path) -> SRDataModule:
    def val(scale):
        return {
            "class_path": "sisr.datasets.pre_upsampled.ValidationDataset",
            "init_args": {"img_dir": str(path), "scale": scale},
        }

    train = {
        "class_path": "sisr.datasets.pre_upsampled.TrainDataset",
        "init_args": {
            "img_dir": str(path),
            "subimg_size": 12,
            "stride": 12,
            "scale": [2, 3],
            "cache_dir": str(path / ".c"),
            "build_num_workers": 1,
        },
    }
    return SRDataModule(
        train_dataset=train,
        val_dataset=val([2, 3]),
        test_datasets={"Set5": val([2, 3])},
        train_dataloader_kwargs={"batch_size": 4, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


class _GradProbe(lightning.Callback):
    def __init__(self):
        self.max_grad = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        grads = [p.grad.abs().max() for p in pl_module.parameters() if p.grad is not None]
        self.max_grad = max(self.max_grad, float(max(grads)))


def _fit(lit, dm, probe=None, **trainer_kwargs):
    trainer = lightning.Trainer(
        max_steps=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        val_check_interval=2,
        check_val_every_n_epoch=None,
        callbacks=[BenchmarkImageLogger(every_n_val_runs=100), *([probe] if probe else [])],
        **trainer_kwargs,
    )
    trainer.fit(lit, datamodule=dm)
    return trainer


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_a_multi_scale_fit_logs_per_scale_tags_and_the_mean_headline(tiny_rgb_image_dir: Path):
    trainer = _fit(_lit(), _dm(tiny_rgb_image_dir))

    m = trainer.callback_metrics
    assert {"psnr/val/Y", "psnr/val/Y/x2", "psnr/val/Y/x3", "loss/val/x2", "loss/val/x3"} <= set(m)
    assert float(m["psnr/val/Y"]) == pytest.approx(
        (float(m["psnr/val/Y/x2"]) + float(m["psnr/val/Y/x3"])) / 2, rel=1e-5
    )
    assert {"psnr/Set5_x2/Y", "psnr/Set5_x3/Y"} <= set(m)


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
@pytest.mark.parametrize("scaled, bound", [(True, 0.002), (False, 0.001)])
def test_scale_clip_with_lr_clips_to_theta_over_the_learning_rate(
    tiny_rgb_image_dir: Path, scaled: bool, bound: float
):
    """lr 0.5 and theta 0.001: VDSR's adjustable clipping bounds every gradient by
    theta / lr = 0.002; without the flag Lightning's plain value clip bounds it by 0.001."""
    probe = _GradProbe()

    _fit(
        _lit(scale_clip_with_lr=scaled),
        _dm(tiny_rgb_image_dir),
        probe,
        gradient_clip_val=0.001,
        gradient_clip_algorithm="value",
    )

    assert probe.max_grad == pytest.approx(bound, rel=1e-4)


def test_predict_rgb_keeps_the_input_size():
    lit = _lit()
    lr = hr = torch.rand(1, 3, 12, 12)

    sr, hr_cropped = lit.predict_rgb(lr, hr)

    assert sr.shape == lr.shape == hr_cropped.shape
