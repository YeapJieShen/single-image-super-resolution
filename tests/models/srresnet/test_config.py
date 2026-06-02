"""Defaults, subclass, and isolation tests for SRResNet's paper-faithful configs."""
from sisr.models.srresnet import SRResNetEvalConfig, SRResNetTrainingConfig
from sisr.training import SREvalConfig, SRTrainingConfig


def test_srresnet_training_config_paper_defaults():
    """SRResNet ships init_strategy='default' (reserved-but-no-op)."""
    cfg = SRResNetTrainingConfig()
    # Paper-faithful Kaiming/PReLU init is future work; current default is PyTorch's built-in init.
    assert cfg.init_strategy == "default"
    assert cfg.init_mean == 0.0
    assert cfg.init_std == 0.01
    assert cfg.layer_lrs is None              # SRResNet has BatchNorm/PReLU; layer_lrs would error


def test_srresnet_eval_config_paper_defaults():
    """SRResNet paper defaults: x4 border crop; RGB+YCbCr PSNR breakdown."""
    cfg = SRResNetEvalConfig()
    assert cfg.crop_border == 4
    assert cfg.psnr_channels == ["RGB", "YCbCr"]
    assert cfg.separate_psnr is False


def test_srresnet_training_config_subclass():
    assert issubclass(SRResNetTrainingConfig, SRTrainingConfig)


def test_srresnet_eval_config_subclass():
    assert issubclass(SRResNetEvalConfig, SREvalConfig)


def test_srresnet_eval_config_psnr_channels_independent_per_instance():
    """`default_factory` produces a fresh list per instance — guards against mutable-default bug."""
    a = SRResNetEvalConfig()
    b = SRResNetEvalConfig()
    a.psnr_channels.append("X")
    assert b.psnr_channels == ["RGB", "YCbCr"]
