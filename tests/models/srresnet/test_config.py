"""Defaults, subclass, and isolation tests for SRResNet's paper-faithful configs."""

import pytest

from sisr.models.srresnet import SRResNetEvalConfig, SRResNetTrainingConfig
from sisr.models.srresnet.model import SRResNet
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import SREvalConfig, SRTrainingConfig


def test_srresnet_training_config_paper_defaults():
    """SRResNet ships init_strategy='default' (reserved-but-no-op)."""
    cfg = SRResNetTrainingConfig()
    # Paper-faithful Kaiming/PReLU init is future work; current default is PyTorch's built-in init.
    assert cfg.init_strategy == "default"
    assert cfg.init_mean == 0.0
    assert cfg.init_std == 0.01
    assert cfg.layer_lrs is None  # SRResNet has BatchNorm/PReLU; layer_lrs would error


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


# ---------------------------------------------------------------------------
# validate_against (INIT.16) — SRResNet-specific in_out_channels/processor check
# ---------------------------------------------------------------------------


def test_srresnet_validate_against_rejects_in_out_channels_mismatch():
    model = SRResNet(scale=2, num_residual_blocks=1, in_out_channels=3)
    with pytest.raises(ValueError, match="in_out_channels"):
        SRResNetTrainingConfig().validate_against(model, YChannelProcessor())


def test_srresnet_validate_against_accepts_matching_in_out_channels():
    model = SRResNet(scale=2, num_residual_blocks=1, in_out_channels=3)
    SRResNetTrainingConfig().validate_against(model, RGBProcessor())  # must not raise


def test_srresnet_validate_against_still_runs_base_forward_probe():
    """SRResNetTrainingConfig.validate_against must chain to the base check
    (example_input_shape/forward probe) via super(), not just its own."""
    model = SRResNet(scale=2, num_residual_blocks=1, in_out_channels=3)
    cfg = SRResNetTrainingConfig(example_input_shape=(1, 16, 16))  # 1 != model_channels=3
    with pytest.raises(ValueError, match="example_input_shape"):
        cfg.validate_against(model, RGBProcessor())
