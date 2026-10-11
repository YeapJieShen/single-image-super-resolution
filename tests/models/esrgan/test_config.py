import pytest

from sisr.models.esrgan import (
    ESRGANEvalConfig,
    ESRGANTrainingConfig,
    RRDBNet,
    RRDBNetEvalConfig,
    RRDBNetTrainingConfig,
)
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training.config import AdversarialTrainingConfig, SREvalConfig, SRTrainingConfig


def test_psnr_stage_defaults():
    cfg = RRDBNetTrainingConfig()

    assert issubclass(RRDBNetTrainingConfig, SRTrainingConfig)
    assert cfg.scale == 4 and cfg.init_strategy == "paper"


def test_gan_stage_defaults_use_the_papers_lambda():
    cfg = ESRGANTrainingConfig()

    assert issubclass(ESRGANTrainingConfig, AdversarialTrainingConfig)
    assert cfg.adversarial_weight == 5e-3 and cfg.scale == 4 and cfg.init_from is None


def test_eval_defaults_and_independent_lists():
    cfg = RRDBNetEvalConfig()

    assert issubclass(RRDBNetEvalConfig, SREvalConfig)
    assert cfg.crop_border is None and cfg.psnr_channels == ["Y"] and cfg.ssim_impl == "wang"
    cfg.psnr_channels.append("X")
    assert RRDBNetEvalConfig().psnr_channels == ["Y"]
    assert ESRGANEvalConfig().perceptual_metrics == ["lpips", "dists"]
    assert RRDBNetEvalConfig().perceptual_metrics == []


@pytest.mark.parametrize("cls", [RRDBNetTrainingConfig, ESRGANTrainingConfig])
def test_validate_against_checks_channels_and_scale(cls):
    model = RRDBNet(features=8, growth=4, num_blocks=1)

    cls(example_input_shape=(3, 8, 8)).validate_against(model, RGBProcessor())
    with pytest.raises(ValueError, match="in_out_channels=3"):
        cls().validate_against(model, YChannelProcessor())
    with pytest.raises(ValueError, match="does not match"):
        cls(scale=2).validate_against(model, RGBProcessor())
