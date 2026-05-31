from sisr.models.srcnn import SRCNNEvalConfig, SRCNNTrainingConfig
from sisr.training import SREvalConfig, SRTrainingConfig


def test_srcnn_training_config_paper_defaults():
    cfg = SRCNNTrainingConfig()
    assert cfg.model_colorspace == "Y"
    assert cfg.layer_lrs == [1.0e-4, 1.0e-4, 1.0e-5]


def test_srcnn_eval_config_paper_defaults():
    cfg = SRCNNEvalConfig()
    assert cfg.crop_border == 3
    assert cfg.psnr_channels == ["RGB", "YCbCr"]
    assert cfg.separate_psnr is False


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
    assert b.psnr_channels == ["RGB", "YCbCr"]


def test_srcnn_training_config_init_defaults():
    """Paper-faithful init lives on SRCNNTrainingConfig after migration."""
    cfg = SRCNNTrainingConfig()
    assert cfg.init_strategy == "paper"
    assert cfg.init_mean == 0.0
    assert cfg.init_std == 0.01
