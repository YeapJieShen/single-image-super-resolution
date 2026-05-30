from dataclasses import fields

from sisr.training import SREvalConfig, SRTrainingConfig


def test_sr_training_config_defaults():
    cfg = SRTrainingConfig()
    assert cfg.model_colorspace == "RGB"
    assert cfg.layer_lrs is None
    assert cfg.example_input_shape is None


def test_sr_eval_config_defaults():
    cfg = SREvalConfig()
    assert cfg.crop_border == 0
    assert cfg.psnr_channels == ["RGB"]
    assert cfg.separate_psnr is False


def test_sr_eval_config_psnr_channels_isolated_per_instance():
    """`field(default_factory=...)` guards against shared mutable defaults."""
    a = SREvalConfig()
    b = SREvalConfig()
    a.psnr_channels.append("YCbCr")
    assert b.psnr_channels == ["RGB"], "default_factory must produce a fresh list per instance"


def test_sr_training_config_field_names():
    """Reduced surface check — guards against accidental field renames."""
    names = {f.name for f in fields(SRTrainingConfig)}
    assert names == {"model_colorspace", "layer_lrs", "example_input_shape"}


def test_sr_eval_config_field_names():
    names = {f.name for f in fields(SREvalConfig)}
    assert names == {"crop_border", "psnr_channels", "separate_psnr"}
