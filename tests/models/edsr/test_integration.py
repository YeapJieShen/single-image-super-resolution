"""EDSR through the real Lightning module: a step, the EDSR+ self-ensemble, and the template."""

from pathlib import Path

import pytest
import torch
import yaml

from sisr.models.edsr import EDSR, EDSREvalConfig, EDSRTrainingConfig
from sisr.processors import RGBProcessor
from sisr.training import SRLightning


def _lit(scale=2) -> SRLightning:
    torch.manual_seed(0)
    return SRLightning(
        model=EDSR(scale=scale, features=8, num_blocks=2),
        processor=RGBProcessor(),
        training_config=EDSRTrainingConfig(scale=scale, example_input_shape=(3, 8, 8)),
        eval_config=EDSREvalConfig(),
        criterion=torch.nn.L1Loss(),
    )


def test_a_training_step_is_finite_and_reaches_every_parameter():
    lit = _lit()
    g = torch.Generator().manual_seed(1)
    lr, hr = torch.rand(2, 3, 8, 8, generator=g), torch.rand(2, 3, 16, 16, generator=g)

    loss = lit._step((lr, hr), need_sr_rgb=False)[0]
    loss.backward()

    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lit.model.parameters())


def test_edsr_plus_is_the_mean_of_the_eight_geometric_transforms():
    """The paper's EDSR+: flip/rotate the LR eight ways, undo each on the output, average.
    Compared with a hand-rolled average, not with the library's own transform helpers."""
    lit = _lit().eval()
    lr = torch.rand(1, 3, 6, 9)
    hr = torch.rand(1, 3, 12, 18)

    with torch.no_grad():
        got, _ = lit.predict_rgb(lr, hr, self_ensemble=True)
        outs = []
        for flip_w in (False, True):
            for flip_h in (False, True):
                for transpose in (False, True):
                    x = lr.flip(-1) if flip_w else lr
                    x = x.flip(-2) if flip_h else x
                    x = x.transpose(-2, -1) if transpose else x
                    y = lit.model(x)
                    y = y.transpose(-2, -1) if transpose else y
                    y = y.flip(-2) if flip_h else y
                    y = y.flip(-1) if flip_w else y
                    outs.append(y)
        expected = torch.stack(outs).mean(0).clamp(0, 1)

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_the_plain_forward_differs_from_the_self_ensemble():
    lit = _lit().eval()
    lr, hr = torch.rand(1, 3, 6, 9), torch.rand(1, 3, 12, 18)

    with torch.no_grad():
        plain, _ = lit.predict_rgb(lr, hr)
        plus, _ = lit.predict_rgb(lr, hr, self_ensemble=True)

    assert not torch.allclose(plain, plus, atol=1e-6)


_TEMPLATE = Path(__file__).resolve().parents[3] / "templates" / "config.edsr.template.yaml"


def test_the_template_carries_the_papers_recipe():
    cfg = yaml.safe_load(_TEMPLATE.read_text("ascii"))
    model = cfg["model"]["init_args"]["model"]["init_args"]

    assert (model["num_blocks"], model["features"], model["residual_scale"]) == (32, 256, 0.1)
    assert cfg["optimizer"]["init_args"]["lr"] == 1.0e-4
    assert cfg["lr_scheduler"]["init_args"] == {"step_size": 200000, "gamma": 0.5}
    assert cfg["model"]["init_args"]["criterion"]["class_path"] == "torch.nn.L1Loss"
    assert cfg["data"]["train_dataloader_kwargs"]["batch_size"] == 16
    assert cfg["data"]["train_dataset"]["init_args"]["hr_crop_size"] == 96  # 48x48 LR at x2


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_the_eval_config_shaves_the_scale_resolved_from_the_model(scale):
    assert _lit(scale).eval_config.crop_border == scale
