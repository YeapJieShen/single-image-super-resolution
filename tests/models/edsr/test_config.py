import pytest

from sisr.models.edsr import EDSR, EDSREvalConfig, EDSRTrainingConfig
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training.config import SREvalConfig, SRTrainingConfig


def test_training_defaults():
    cfg = EDSRTrainingConfig()

    assert issubclass(EDSRTrainingConfig, SRTrainingConfig)
    assert cfg.scale == 2 and cfg.init_strategy == "default"


def test_eval_defaults_and_independent_lists():
    cfg = EDSREvalConfig()

    assert issubclass(EDSREvalConfig, SREvalConfig)
    assert cfg.crop_border is None and cfg.psnr_channels == ["Y"] and cfg.ssim_impl == "wang"
    cfg.psnr_channels.append("X")
    assert EDSREvalConfig().psnr_channels == ["Y"]


def test_validate_against_checks_channels_and_scale():
    model = EDSR(scale=3, features=4, num_blocks=1)

    EDSRTrainingConfig(scale=3, example_input_shape=(3, 8, 8)).validate_against(
        model, RGBProcessor()
    )
    with pytest.raises(ValueError, match="in_out_channels=3"):
        EDSRTrainingConfig(scale=3).validate_against(model, YChannelProcessor())
    with pytest.raises(ValueError, match="does not match"):
        EDSRTrainingConfig(scale=2).validate_against(model, RGBProcessor())
