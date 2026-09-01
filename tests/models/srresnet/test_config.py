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
    assert cfg.scale == 4  # Ledig et al. §3.2's SRResNet baseline is fixed at 4x


def test_srresnet_eval_config_paper_defaults():
    """SRResNet paper defaults: x4 border crop; RGB+Y PSNR.

    Regression: defaults must report the paper's own Y-channel
    metric, not the YCbCr 3-channel aggregate (which reads optimistically
    high since chroma planes are far smoother than luma)."""
    cfg = SRResNetEvalConfig()
    # None = derive the field convention's `scale` pixels; SRLightning resolves it.
    # Previously a hardcoded 4, right only because this template ships x4.
    assert cfg.crop_border is None
    assert cfg.psnr_channels == ["RGB", "Y"]
    assert cfg.separate_psnr is False
    # Not overridden — inherits the base SREvalConfig default.
    assert cfg.ssim_channels == ["RGB", "Y"]


def test_srresnet_training_config_subclass():
    assert issubclass(SRResNetTrainingConfig, SRTrainingConfig)


def test_srresnet_eval_config_subclass():
    assert issubclass(SRResNetEvalConfig, SREvalConfig)


def test_srresnet_eval_config_defaults_to_daala_ssim():
    """Ledig et al. computed SSIM with the daala package, so the paper-faithful
    default belongs here — the same place crop_border=4 lives. The base
    SREvalConfig stays on 'wang', so SRCNN is unaffected."""
    assert SRResNetEvalConfig().ssim_impl == "daala"
    assert SREvalConfig().ssim_impl == "wang"


def test_srresnet_eval_config_psnr_channels_independent_per_instance():
    """`default_factory` produces a fresh list per instance — guards against mutable-default bug."""
    a = SRResNetEvalConfig()
    b = SRResNetEvalConfig()
    a.psnr_channels.append("X")
    assert b.psnr_channels == ["RGB", "Y"]


# ---------------------------------------------------------------------------
# validate_against — SRResNet-specific in_out_channels/processor check
# ---------------------------------------------------------------------------


def test_srresnet_validate_against_rejects_in_out_channels_mismatch():
    model = SRResNet(scale=2, num_residual_blocks=1, in_out_channels=3)
    with pytest.raises(ValueError, match="in_out_channels"):
        SRResNetTrainingConfig().validate_against(model, YChannelProcessor())


def test_srresnet_validate_against_accepts_matching_in_out_channels():
    # scale=4 matches SRResNetTrainingConfig's paper-fixed default; the point
    # of this test is the channel check, not scale.
    model = SRResNet(scale=4, num_residual_blocks=1, in_out_channels=3)
    SRResNetTrainingConfig().validate_against(model, RGBProcessor())  # must not raise


def test_srresnet_validate_against_still_runs_base_forward_probe():
    """SRResNetTrainingConfig.validate_against must chain to the base check
    (example_input_shape/forward probe) via super(), not just its own."""
    # scale=4 matches the default config's scale so only the intended
    # example_input_shape channel mismatch fires.
    model = SRResNet(scale=4, num_residual_blocks=1, in_out_channels=3)
    cfg = SRResNetTrainingConfig(example_input_shape=(1, 16, 16))  # 1 != model_channels=3
    with pytest.raises(ValueError, match="example_input_shape"):
        cfg.validate_against(model, RGBProcessor())


def test_srresnet_validate_against_rejects_scale_mismatch():
    """Regression: SRResNetTrainingConfig's paper-fixed scale=4 default must
    actually be checked, not merely recorded — pairing it with a
    differently-scaled model is a real misconfiguration."""
    model = SRResNet(scale=2, num_residual_blocks=1)
    with pytest.raises(ValueError, match="scale"):
        SRResNetTrainingConfig().validate_against(model, RGBProcessor())
