"""MDSR through the real Lightning module: per-batch scales, per-scale tags, the template."""

from pathlib import Path

import lightning
import pytest
import torch
import yaml

from sisr.models.mdsr import MDSR, MDSREvalConfig, MDSRTrainingConfig
from sisr.processors import RGBProcessor
from sisr.training import BenchmarkImageLogger, SRDataModule, SRLightning


def _lit(scales=(2, 3)) -> SRLightning:
    torch.manual_seed(0)
    return SRLightning(
        model=MDSR(scale=scales, features=8, num_blocks=2),
        processor=RGBProcessor(),
        training_config=MDSRTrainingConfig(scale=list(scales)),
        eval_config=MDSREvalConfig(ssim_channels=[]),
        criterion=torch.nn.L1Loss(),
        optimizer=lambda p: torch.optim.Adam(p, lr=1e-3),
    )


def _dm(path: Path) -> SRDataModule:
    def val(scale):
        return {
            "class_path": "sisr.datasets.native_lr.ValidationDataset",
            "init_args": {"img_dir": str(path), "scale": scale},
        }

    return SRDataModule(
        train_dataset={
            "class_path": "sisr.datasets.native_lr.TrainDataset",
            "init_args": {
                "img_dir": str(path),
                "scale": [2, 3],
                "hr_crop_size": 12,
                "crops_per_image": 8,
                "cache_dir": str(path / ".c"),
                "build_num_workers": 1,
            },
        },
        val_dataset=val([2, 3]),
        test_datasets={"Set5": val([2, 3])},
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


class _SeenScales(lightning.Callback):
    def __init__(self):
        self.scales = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        lr, hr = batch
        self.scales.append(hr.shape[-1] // lr.shape[-1])


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_a_fit_draws_a_scale_per_batch_and_logs_per_scale_tags(tiny_rgb_image_dir: Path):
    lit = _lit()
    seen = _SeenScales()
    trainer = lightning.Trainer(
        max_steps=12,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        val_check_interval=12,
        check_val_every_n_epoch=None,
        callbacks=[BenchmarkImageLogger(every_n_val_runs=100), seen],
    )
    torch.manual_seed(0)

    trainer.fit(lit, datamodule=_dm(tiny_rgb_image_dir))

    assert set(seen.scales) == {2, 3}
    m = trainer.callback_metrics
    assert {
        "psnr/val/Y",
        "psnr/val/Y/x2",
        "psnr/val/Y/x3",
        "psnr/Set5_x2/Y",
        "psnr/Set5_x3/Y",
    } <= set(m)


def test_the_module_hands_each_batch_its_scale_to_the_model():
    lit = _lit()
    seen = []
    lit.model.register_forward_pre_hook(
        lambda m, args, kwargs: seen.append(kwargs.get("scale")), with_kwargs=True
    )

    for s in (2, 3):
        lr = torch.rand(1, 3, 4, 4)
        lit._step((lr, torch.rand(1, 3, 4 * s, 4 * s)), need_sr_rgb=False)

    assert seen == [2, 3]


def test_predict_needs_a_predict_scale_when_several_are_configured():
    lit = _lit()

    with pytest.raises(ValueError, match="predict_scale"):
        lit.predict_step(torch.rand(1, 3, 4, 4), 0)


_TEMPLATE = Path(__file__).resolve().parents[3] / "templates" / "config.mdsr.template.yaml"


def test_the_template_carries_the_papers_recipe():
    cfg = yaml.safe_load(_TEMPLATE.read_text("ascii"))
    model = cfg["model"]["init_args"]["model"]["init_args"]

    assert model["scale"] == [2, 3, 4] and model["num_blocks"] == 80 and model["features"] == 64
    train = cfg["model"]["init_args"]["training_config"]["init_args"]
    assert train["scale"] == [2, 3, 4]
    assert cfg["optimizer"]["init_args"]["lr"] == 1.0e-4
    assert cfg["lr_scheduler"]["init_args"] == {"step_size": 200000, "gamma": 0.5}
    assert cfg["model"]["init_args"]["criterion"]["class_path"] == "torch.nn.L1Loss"
    assert cfg["data"]["train_dataloader_kwargs"]["batch_size"] == 16
    assert cfg["data"]["train_dataset"]["init_args"]["scale"] == [2, 3, 4]
