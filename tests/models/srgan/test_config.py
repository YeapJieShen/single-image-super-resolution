"""Defaults, subclass, and validation tests for SRGAN's paper-faithful configs."""

import pytest

from sisr.models.srgan import SRGANEvalConfig, SRGANTrainingConfig
from sisr.models.srresnet import SRResNetEvalConfig, SRResNetTrainingConfig


def test_paper_defaults():
    cfg = SRGANTrainingConfig()
    assert cfg.adversarial_weight == pytest.approx(1e-3)
    assert cfg.d_steps_per_g_step == 1  # Ledig's 1:1 alternation
    assert cfg.scale == 4  # inherited from SRResNetTrainingConfig
    assert cfg.init_from is None  # optional: unset trains from scratch


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="d_steps_per_g_step"):
        SRGANTrainingConfig(d_steps_per_g_step=0)


def test_cuda_graph_refused_at_construction():
    """Manual optimization with two alternating optimizers cannot be captured."""
    with pytest.raises(ValueError, match="cuda_graph"):
        SRGANTrainingConfig(cuda_graph=True)


def test_eval_config_turns_perceptual_metrics_on():
    """PSNR/SSIM get worse by design here, so the run needs a metric that means
    something."""
    cfg = SRGANEvalConfig()
    assert cfg.perceptual_keys == ["lpips", "dists"]
    assert cfg.ssim_impl == "daala"  # inherited
    assert cfg.crop_border == 4  # inherited


def test_training_config_subclass():
    assert issubclass(SRGANTrainingConfig, SRResNetTrainingConfig)


def test_eval_config_subclass():
    assert issubclass(SRGANEvalConfig, SRResNetEvalConfig)


def test_cuda_graph_and_compile_backend_both_set_reports_base_error_first():
    """super().__post_init__() runs before the SRGAN-only checks, so the inherited
    cuda_graph/compile_backend conflict (not the SRGAN cuda_graph-is-unsupported
    message) is what surfaces when both are set."""
    with pytest.raises(ValueError, match="compile_backend"):
        SRGANTrainingConfig(cuda_graph=True, compile_backend="inductor")


def test_eval_config_perceptual_metrics_independent_per_instance():
    """`default_factory` produces a fresh list per instance — guards against a
    mutable-default bug (the same class of bug SRResNet's psnr_channels test guards)."""
    a = SRGANEvalConfig()
    b = SRGANEvalConfig()
    a.perceptual_metrics.append("x")
    assert b.perceptual_keys == ["lpips", "dists"]
