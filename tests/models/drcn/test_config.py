import pytest

from sisr.models.drcn import DRCN, DRCNEvalConfig, DRCNTrainingConfig
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import SREvalConfig, SRTrainingConfig


def test_training_config_defaults():
    cfg = DRCNTrainingConfig()
    assert issubclass(DRCNTrainingConfig, SRTrainingConfig)
    assert cfg.init_strategy == "paper"
    assert cfg.scale is None


def test_eval_config_defaults_and_independent_lists():
    cfg = DRCNEvalConfig()
    assert issubclass(DRCNEvalConfig, SREvalConfig)
    assert cfg.crop_border is None and cfg.psnr_channels == ["Y"]
    cfg.psnr_channels.append("X")
    assert DRCNEvalConfig().psnr_channels == ["Y"]


def test_validate_against_accepts_matching_and_rejects_mismatched_channels():
    model = DRCN(in_out_channels=1, features=4, recursions=2)
    DRCNTrainingConfig(example_input_shape=(1, 8, 8)).validate_against(model, YChannelProcessor())
    with pytest.raises(ValueError, match="in_out_channels=1"):
        DRCNTrainingConfig().validate_against(model, RGBProcessor())
