"""StudioYChannelProcessor: the model sees MATLAB studio-range Y; the returned RGB is exact."""

import pytest
import torch
import torch.nn.functional as F

from sisr.colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from sisr.models.vdsr import VDSR, VDSRTrainingConfig
from sisr.processors import SRProcessor, StudioYChannelProcessor, YChannelProcessor


def test_properties():
    p = StudioYChannelProcessor()
    assert isinstance(p, SRProcessor) and not isinstance(p, YChannelProcessor)
    assert p.model_channels == 1
    assert p.output_range == pytest.approx((16 / 255, 235 / 255))
    assert p.output_colorspace == "Y"


def test_extract_gives_studio_luma_at_the_endpoints():
    p = StudioYChannelProcessor()
    black, white = torch.zeros(1, 3, 4, 4), torch.ones(1, 3, 4, 4)
    assert p.extract(black).shape == (1, 1, 4, 4)
    assert torch.allclose(p.extract(black), torch.tensor(16 / 255), atol=1e-6)
    assert torch.allclose(p.extract(white), torch.tensor(235 / 255), atol=1e-6)
    x = torch.rand(2, 3, 4, 4)
    torch.testing.assert_close(p.extract_target(x), p.extract(x))


def test_reconstruct_of_extract_returns_the_original_rgb():
    p = StudioYChannelProcessor()
    x = torch.rand(2, 3, 16, 16)
    torch.testing.assert_close(p.reconstruct(p.extract(x), x), x, atol=2e-5, rtol=0)


def test_reconstruct_upscaled_maps_back_to_full_y_and_stitches_bicubic_chroma():
    p = StudioYChannelProcessor()
    lr = torch.rand(2, 3, 8, 8)
    sr_y = 16 / 255 + (219 / 255) * torch.rand(2, 1, 32, 32)
    out = p.reconstruct(sr_y, lr)
    assert out.shape == (2, 3, 32, 32)
    full_y = (sr_y - 16 / 255) * 255 / 219
    cbcr = F.interpolate(rgb_to_ycbcr(lr)[:, 1:], size=(32, 32), mode="bicubic")
    torch.testing.assert_close(out, ycbcr_to_rgb(torch.cat([full_y, cbcr], 1)), atol=1e-5, rtol=0)


def test_a_one_channel_model_pairs_with_it_and_a_three_channel_one_does_not():
    cfg = VDSRTrainingConfig(example_input_shape=(1, 8, 8))
    cfg.validate_against(VDSR(in_out_channels=1, depth=3, features=4), StudioYChannelProcessor())
    with pytest.raises(ValueError, match="model_channels"):
        cfg.validate_against(
            VDSR(in_out_channels=3, depth=3, features=4), StudioYChannelProcessor()
        )


@pytest.mark.parametrize("name", ["vdsr", "drcn"])
def test_the_matlab_trained_templates_use_the_studio_y_processor(name):
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "templates" / f"config.{name}.template.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="ascii"))
    assert cfg["model"]["init_args"]["processor"]["class_path"] == (
        "sisr.processors.StudioYChannelProcessor"
    )
