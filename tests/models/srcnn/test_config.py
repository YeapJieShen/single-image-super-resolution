import pytest

from sisr.models.srcnn import SRCNN, SRCNNEvalConfig, SRCNNTrainingConfig
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import SREvalConfig, SRTrainingConfig


def test_srcnn_training_config_paper_defaults():
    cfg = SRCNNTrainingConfig()
    # model_colorspace was removed in the SR base-classes refactor;
    # colorspace intent is now expressed by pairing with sisr.processors.YChannelProcessor.
    assert cfg.layer_lrs == [1.0e-4, 1.0e-4, 1.0e-5]


def test_srcnn_eval_config_paper_defaults():
    """Regression (P2.9): defaults must report the paper's own Y-channel
    metric, not the YCbCr 3-channel aggregate (which reads optimistically
    high since chroma planes are far smoother than luma)."""
    cfg = SRCNNEvalConfig()
    assert cfg.crop_border == 3
    assert cfg.psnr_channels == ["RGB", "Y"]
    assert cfg.separate_psnr is False
    # Not overridden — inherits the base SREvalConfig default.
    assert cfg.ssim_channels == ["RGB", "Y"]


def test_srcnn_training_config_subclass():
    assert issubclass(SRCNNTrainingConfig, SRTrainingConfig)


def test_srcnn_eval_config_subclass():
    assert issubclass(SRCNNEvalConfig, SREvalConfig)


def test_srcnn_training_config_layer_lrs_independent_per_instance():
    """`default_factory` must produce a fresh list per instance — guards
    against the classic mutable-default trap."""
    a = SRCNNTrainingConfig()
    b = SRCNNTrainingConfig()
    a.layer_lrs.append(1.0)
    assert b.layer_lrs == [1.0e-4, 1.0e-4, 1.0e-5]


def test_srcnn_eval_config_psnr_channels_independent_per_instance():
    a = SRCNNEvalConfig()
    b = SRCNNEvalConfig()
    a.psnr_channels.append("X")
    assert b.psnr_channels == ["RGB", "Y"]


def test_srcnn_training_config_init_defaults():
    """Paper-faithful init lives on SRCNNTrainingConfig after migration.

    init_std=0.001 per Dong et al. §Training; SRCNNTrainingConfig overrides
    the shared base's 0.01 (which is not itself a paper value)."""
    cfg = SRCNNTrainingConfig()
    assert cfg.init_strategy == "paper"
    assert cfg.init_mean == 0.0
    assert cfg.init_std == 0.001


# ---------------------------------------------------------------------------
# validate_against (INIT.16) — SRCNN-specific num_channels/processor check
# ---------------------------------------------------------------------------


def test_srcnn_validate_against_rejects_num_channels_mismatch():
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    with pytest.raises(ValueError, match="num_channels"):
        SRCNNTrainingConfig().validate_against(model, YChannelProcessor())


def test_srcnn_validate_against_accepts_matching_num_channels():
    model = SRCNN(num_channels=1, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    SRCNNTrainingConfig().validate_against(model, YChannelProcessor())  # must not raise


def test_srcnn_validate_against_still_runs_base_forward_probe():
    """SRCNNTrainingConfig.validate_against must chain to the base check
    (example_input_shape/forward probe) via super(), not just its own."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    cfg = SRCNNTrainingConfig(example_input_shape=(1, 33, 33))  # 1 != processor.model_channels=3
    with pytest.raises(ValueError, match="example_input_shape"):
        cfg.validate_against(model, RGBProcessor())
