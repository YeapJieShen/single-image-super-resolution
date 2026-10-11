import pytest

from sisr.models.vdsr import VDSR, VDSREvalConfig, VDSRTrainingConfig
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import SREvalConfig, SRTrainingConfig


def test_training_config_defaults():
    cfg = VDSRTrainingConfig()

    assert issubclass(VDSRTrainingConfig, SRTrainingConfig)
    assert cfg.init_strategy == "paper" and cfg.scale is None and cfg.scale_clip_with_lr is False


def test_training_config_takes_a_scale_list_or_an_int():
    assert VDSRTrainingConfig(scale=[2, 3, 4]).scale == [2, 3, 4]
    assert VDSRTrainingConfig(scale=[3]).scale == 3
    assert VDSRTrainingConfig(scale=2).scale == 2


def test_eval_config_defaults_and_independent_lists():
    cfg = VDSREvalConfig()

    assert issubclass(VDSREvalConfig, SREvalConfig)
    assert cfg.crop_border is None and cfg.psnr_channels == ["Y"]
    cfg.psnr_channels.append("X")
    assert VDSREvalConfig().psnr_channels == ["Y"]


def test_validate_against_accepts_matching_and_rejects_mismatched_channels():
    model = VDSR(in_out_channels=1, depth=3, features=4)
    cfg = VDSRTrainingConfig(scale=[2, 3], example_input_shape=(1, 8, 8))

    cfg.validate_against(model, YChannelProcessor())  # a pre-upsampled model has no scale hparam
    with pytest.raises(ValueError, match="in_out_channels=1"):
        VDSRTrainingConfig().validate_against(model, RGBProcessor())
