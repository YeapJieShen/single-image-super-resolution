"""ESRGAN through the real adversarial module: RRDBNet, VGG-style critic, relativistic loss."""

from pathlib import Path

import lightning
import pytest
import torch
import yaml

from sisr import artifacts
from sisr.losses import RelativisticAverageLoss, VGG19FeatureLoss, WeightedSumLoss
from sisr.models.esrgan import (
    ESRGANEvalConfig,
    ESRGANTrainingConfig,
    RRDBNet,
    RRDBNetEvalConfig,
    RRDBNetTrainingConfig,
    VGGStyleDiscriminator,
)
from sisr.processors import RGBProcessor
from sisr.training import SRLightning
from sisr.training.gan_module import SRGANLightning
from sisr.training.metadata import build_metadata

pytestmark = pytest.mark.filterwarnings("ignore:VGG19FeatureLoss.weights=None.:UserWarning")


def _net():
    return RRDBNet(features=8, growth=4, num_blocks=2)


class _Probe(RelativisticAverageLoss):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generator_loss(self, logits_real, logits_fake):
        self.calls.append((logits_real.requires_grad, logits_fake.requires_grad))
        return super().generator_loss(logits_real, logits_fake)


def _gan(adversarial_loss, init_from=None):
    torch.manual_seed(0)
    return SRGANLightning(
        model=_net(),
        processor=RGBProcessor(),
        discriminator=VGGStyleDiscriminator(hr_input_size=32, base_channels=4, dense_features=8),
        adversarial_loss=adversarial_loss,
        criterion=WeightedSumLoss(
            terms={
                "vgg": VGG19FeatureLoss(layer="vgg54", before_activation=True, weights=None),
                "l1": torch.nn.L1Loss(),
            },
            weights={"vgg": 1.0, "l1": 1e-2},
        ),
        training_config=ESRGANTrainingConfig(init_from=init_from, example_input_shape=(3, 8, 8)),
        eval_config=ESRGANEvalConfig(perceptual_metrics=[]),
        optimizer=lambda p: torch.optim.SGD(p, lr=0.05),
        discriminator_optimizer=lambda p: torch.optim.SGD(p, lr=0.05),
    )


def _loader():
    g = torch.Generator().manual_seed(1)
    return torch.utils.data.DataLoader(
        [(torch.rand(2, 3, 8, 8, generator=g), torch.rand(2, 3, 32, 32, generator=g))
         for _ in range(2)],
        batch_size=None,
    )  # fmt: skip


def _fit(module):
    trainer = lightning.Trainer(
        max_steps=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    before_g = [p.detach().clone() for p in module.model.parameters()]
    before_d = [p.detach().clone() for p in module.discriminator.parameters()]
    trainer.fit(module, _loader())
    return trainer, before_g, before_d


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_both_networks_step_under_the_relativistic_loss_with_detached_real_logits():
    probe = _Probe()
    module = _gan(probe)

    trainer, before_g, before_d = _fit(module)

    assert any(
        not torch.equal(a, b) for a, b in zip(before_g, module.model.parameters(), strict=True)
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before_d, module.discriminator.parameters(), strict=True)
    )
    assert probe.calls and all(real is False and fake is True for real, fake in probe.calls)
    m = trainer.callback_metrics
    assert all(torch.isfinite(m[k]) for k in ("loss/train", "loss/train/adv", "loss/train/d"))
    assert set(module.criterion.last_terms) == {"vgg", "l1"}


def test_the_gan_stage_initialises_from_the_psnr_stage_artifact(tmp_path):
    """The PSNR stage's weights artifact is a valid ``init_from`` for the GAN stage."""
    psnr = SRLightning(
        model=_net(),
        processor=RGBProcessor(),
        training_config=RRDBNetTrainingConfig(),
        eval_config=RRDBNetEvalConfig(),
    )
    path = tmp_path / "psnr.safetensors"
    artifacts.save(path, psnr.model.state_dict(), build_metadata(psnr))
    gan = _gan(RelativisticAverageLoss(), init_from=str(path))

    gan.setup("fit")

    for a, b in zip(psnr.model.parameters(), gan.model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_the_critic_size_is_checked_against_the_hr_crop_the_generator_emits():
    module = _gan(RelativisticAverageLoss())

    module._extra_probe(torch.rand(3, 8, 8), torch.rand(3, 32, 32), "train_dataset")  # fits
    with pytest.raises(ValueError, match="32"):
        module._extra_probe(torch.rand(3, 16, 16), torch.rand(3, 64, 64), "train_dataset")


# --- the shipped recipes -----------------------------------------------------------------


_TEMPLATES = Path(__file__).resolve().parents[3] / "templates"


def test_the_psnr_template_carries_the_papers_stage_one_recipe():
    cfg = yaml.safe_load((_TEMPLATES / "config.esrgan_psnr.template.yaml").read_text("ascii"))

    assert cfg["optimizer"]["init_args"]["lr"] == 2.0e-4
    assert cfg["lr_scheduler"]["init_args"] == {"step_size": 200000, "gamma": 0.5}
    assert cfg["model"]["init_args"]["criterion"]["class_path"] == "torch.nn.L1Loss"
    assert cfg["data"]["train_dataloader_kwargs"]["batch_size"] == 16
    assert cfg["data"]["train_dataset"]["init_args"]["hr_crop_size"] == 128


def test_the_gan_template_carries_the_papers_stage_two_recipe():
    cfg = yaml.safe_load((_TEMPLATES / "config.esrgan.template.yaml").read_text("ascii"))
    init = cfg["model"]["init_args"]
    vgg = init["criterion"]["init_args"]["terms"]["vgg"]["init_args"]

    assert vgg["layer"] == "vgg54" and vgg["before_activation"] is True
    assert init["criterion"]["init_args"]["weights"] == {"vgg": 1.0, "l1": 1.0e-2}
    assert init["adversarial_loss"]["class_path"] == "sisr.losses.RelativisticAverageLoss"
    assert cfg["lr_scheduler"]["init_args"]["milestones"] == [50000, 100000, 200000, 300000]
    assert cfg["optimizer"]["init_args"]["lr"] == 1.0e-4
    assert init["training_config"]["class_path"].endswith("ESRGANTrainingConfig")
    assert init["training_config"]["init_args"]["init_from"].endswith(".safetensors")
