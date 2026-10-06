import pytest

from sisr import artifacts
from sisr.models.mdsr import MDSR, MDSREvalConfig, MDSRTrainingConfig
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import SRLightning
from sisr.training.config import SREvalConfig, SRTrainingConfig
from sisr.training.metadata import build_metadata


def test_training_defaults_are_the_multi_scale_list():
    cfg = MDSRTrainingConfig()

    assert issubclass(MDSRTrainingConfig, SRTrainingConfig)
    assert cfg.scale == [2, 3, 4] and cfg.init_strategy == "default"
    cfg.scale.append(8)  # type: ignore[union-attr]
    assert MDSRTrainingConfig().scale == [2, 3, 4]


def test_eval_defaults():
    cfg = MDSREvalConfig()

    assert issubclass(MDSREvalConfig, SREvalConfig)
    assert cfg.crop_border is None and cfg.psnr_channels == ["Y"] and cfg.ssim_impl == "wang"


def test_validate_against_compares_the_list_with_the_models_scales():
    model = MDSR(features=4, num_blocks=1)

    MDSRTrainingConfig(example_input_shape=(3, 8, 8)).validate_against(model, RGBProcessor())
    MDSRTrainingConfig(scale=[4, 2, 3]).validate_against(model, RGBProcessor())  # a set
    with pytest.raises(ValueError, match="does not match"):
        MDSRTrainingConfig(scale=[2, 3]).validate_against(model, RGBProcessor())
    with pytest.raises(ValueError, match="in_out_channels=3"):
        MDSRTrainingConfig().validate_against(model, YChannelProcessor())


def test_artifact_metadata_records_the_scale_list_and_the_filename_carries_it():
    lit = SRLightning(
        model=MDSR(features=4, num_blocks=1),
        processor=RGBProcessor(),
        training_config=MDSRTrainingConfig(),
        eval_config=MDSREvalConfig(),
    )

    meta = build_metadata(lit)

    assert meta["io"]["scale"] == [2, 3, 4]
    assert meta["io"]["crop_border_by_scale"] == {"2": 2, "3": 3, "4": 4}
    assert artifacts.stem(meta) == "MDSR_x2-3-4_RGB_1B4F"
