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


def test_eval_config_turns_perceptual_metrics_on():
    """PSNR/SSIM get worse by design here, so the run needs a metric that means
    something."""
    cfg = SRGANEvalConfig()
    assert cfg.perceptual_keys == ["lpips", "dists"]
    assert cfg.ssim_impl == "daala"  # inherited
    assert cfg.crop_border is None  # inherited: derived from scale at construction


def test_training_config_subclass():
    assert issubclass(SRGANTrainingConfig, SRResNetTrainingConfig)


def test_eval_config_subclass():
    assert issubclass(SRGANEvalConfig, SRResNetEvalConfig)


def test_eval_config_perceptual_metrics_independent_per_instance():
    """`default_factory` produces a fresh list per instance — guards against a
    mutable-default bug (the same class of bug SRResNet's psnr_channels test guards)."""
    a = SRGANEvalConfig()
    b = SRGANEvalConfig()
    a.perceptual_metrics.append("x")
    assert b.perceptual_keys == ["lpips", "dists"]


def test_srgan_training_config_runs_its_parents_post_init():
    """SRGANTrainingConfig.__post_init__ overrode its parent's without calling
    super(), so the compile_mode/compile_backend guard was dead for every SRGAN
    config -- the longest-running configuration in the project, and so the most
    expensive place to lose a startup check. An audit found this the only
    __post_init__ in the package that skipped its parent."""
    with pytest.raises(ValueError, match="compile_mode"):
        SRGANTrainingConfig(compile_mode="max-autotune", compile_backend=None)


def test_srgan_training_config_rejects_negative_adversarial_weight():
    """A negative weight inverts the generator's adversarial objective: it would
    train to look MORE fake. Zero is legitimate -- it is the content-only
    ablation the golden-mse run uses."""
    SRGANTrainingConfig(adversarial_weight=0.0)  # valid -- must not raise
    with pytest.raises(ValueError, match="adversarial_weight"):
        SRGANTrainingConfig(adversarial_weight=-1e-3)


def test_srgan_training_config_keeps_its_own_validation():
    """super() must be added without losing what was already there."""
    with pytest.raises(ValueError, match="d_steps_per_g_step"):
        SRGANTrainingConfig(d_steps_per_g_step=0)
