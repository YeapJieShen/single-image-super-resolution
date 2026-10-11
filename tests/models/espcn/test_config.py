import functools
from unittest.mock import MagicMock

import pytest
import torch

from sisr.models.espcn import ESPCN, ESPCNEvalConfig, ESPCNTrainingConfig
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import SREvalConfig, SRLightning, SRTrainingConfig


def test_training_config_paper_defaults():
    cfg = ESPCNTrainingConfig()
    assert issubclass(ESPCNTrainingConfig, SRTrainingConfig)
    assert cfg.scale == 3
    assert cfg.init_strategy == "default"
    assert cfg.layer_lrs == [1.0e-2, 1.0e-2, 1.0e-3]


def test_final_layer_learns_ten_times_slower():
    lrs = ESPCNTrainingConfig().layer_lrs
    assert lrs is not None and lrs[2] == pytest.approx(lrs[0] / 10) and lrs[0] == lrs[1]


def test_layer_lrs_independent_per_instance():
    a, b = ESPCNTrainingConfig(), ESPCNTrainingConfig()
    assert a.layer_lrs is not None
    a.layer_lrs.append(1.0)
    assert b.layer_lrs == [1.0e-2, 1.0e-2, 1.0e-3]


def test_eval_config_defaults():
    cfg = ESPCNEvalConfig()
    assert issubclass(ESPCNEvalConfig, SREvalConfig)
    assert cfg.crop_border is None
    assert cfg.psnr_channels == ["Y"]
    a, b = ESPCNEvalConfig(), ESPCNEvalConfig()
    a.psnr_channels.append("X")
    assert b.psnr_channels == ["Y"]


def test_validate_against_rejects_channel_mismatch():
    with pytest.raises(ValueError, match="YChannelProcessor"):
        ESPCNTrainingConfig().validate_against(
            ESPCN(scale=3, in_out_channels=3), YChannelProcessor()
        )


def test_validate_against_accepts_match_and_runs_base_probe():
    cfg = ESPCNTrainingConfig(example_input_shape=(1, 12, 12))
    cfg.validate_against(ESPCN(scale=3), YChannelProcessor())
    cfg = ESPCNTrainingConfig(example_input_shape=(3, 12, 12))
    cfg.validate_against(ESPCN(scale=3, in_out_channels=3), RGBProcessor())


def test_validate_against_rejects_scale_mismatch():
    with pytest.raises(ValueError, match="scale"):
        ESPCNTrainingConfig(scale=4).validate_against(ESPCN(scale=3), YChannelProcessor())


def _lit(**overrides) -> SRLightning:
    cfg = ESPCNTrainingConfig(example_input_shape=(1, 12, 12), **overrides)
    return SRLightning(
        model=ESPCN(scale=3),
        processor=YChannelProcessor(),
        training_config=cfg,
        eval_config=ESPCNEvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-2, momentum=0.9),
    )


def _batch():
    g = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 12, 12, generator=g), torch.rand(2, 3, 36, 36, generator=g)


def test_lightning_training_and_validation_step():
    lit = _lit()
    lit.log = MagicMock()  # no Trainer attached
    loss = lit.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    lit.validation_step(_batch(), 0)


def test_layer_lrs_become_three_param_groups_with_last_at_a_tenth():
    opt = _lit().configure_optimizers()
    optimizer = opt["optimizer"] if isinstance(opt, dict) else opt
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert len(lrs) == 3
    assert lrs[2] == pytest.approx(lrs[0] / 10)


def test_without_layer_lrs_there_is_one_group():
    opt = _lit(layer_lrs=None).configure_optimizers()
    optimizer = opt["optimizer"] if isinstance(opt, dict) else opt
    assert len(optimizer.param_groups) == 1
