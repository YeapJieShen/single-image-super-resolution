"""The official-weights port: names (always) and parity with the authors' release (data-gated).

The gated tests need ``data/weights/esrgan/RRDB_PSNR_x4.pth`` (see MANIFEST.md there) and the
benchmark sets. They skip when absent; ``SISR_DATA_DIR`` points them at another checkout's
``data/`` (a worktree has none).
"""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sisr.colorspace import rgb_to_ycbcr_studio
from sisr.datasets.native_lr import ValidationDataset
from sisr.models.esrgan import RRDBNet
from sisr.models.esrgan.port import load_official, rename_key

DATA = Path(os.environ.get("SISR_DATA_DIR") or Path(__file__).resolve().parents[3] / "data")
PSNR_WEIGHTS = DATA / "weights" / "esrgan" / "RRDB_PSNR_x4.pth"
needs_weights = pytest.mark.skipif(not PSNR_WEIGHTS.exists(), reason="official weights absent")


@pytest.mark.parametrize(
    "official, ours",
    [
        ("conv_first.weight", "head.weight"),
        ("RRDB_trunk.3.RDB2.conv4.bias", "trunk.3.rdb2.conv4.bias"),
        ("RRDB_trunk.22.RDB3.conv5.weight", "trunk.22.rdb3.conv5.weight"),
        ("trunk_conv.bias", "trunk_conv.bias"),
        ("upconv1.weight", "up.0.weight"),
        ("upconv2.bias", "up.1.bias"),
        ("HRconv.weight", "hr_conv.weight"),
        ("conv_last.bias", "tail.bias"),
    ],
)
def test_official_names_map_to_ours(official, ours):
    assert rename_key(official) == ours


def test_an_unknown_official_name_is_refused():
    with pytest.raises(ValueError, match="unexpected"):
        rename_key("RRDB_trunk.0.RDB4.conv1.weight")


def test_a_renamed_state_dict_loads_strictly_into_a_small_net(tmp_path):
    net = RRDBNet(features=8, growth=4, num_blocks=2)
    official = {}
    for k, v in net.state_dict().items():
        k = (
            k.replace("head.", "conv_first.")
            .replace("hr_conv.", "HRconv.")
            .replace("tail.", "conv_last.")
        )
        k = k.replace("up.0.", "upconv1.").replace("up.1.", "upconv2.")
        for i in "0123456789":
            k = k.replace(f"trunk.{i}.rdb", f"RRDB_trunk.{i}.RDB")
        official[k] = v
    path = tmp_path / "x.pth"
    torch.save(official, path)

    RRDBNet(features=8, growth=4, num_blocks=2).load_state_dict(load_official(path), strict=True)


@needs_weights
def test_every_official_tensor_lands_on_a_parameter_of_the_same_shape():
    state = load_official(PSNR_WEIGHTS)
    ours = RRDBNet().state_dict()

    assert state.keys() == ours.keys()
    assert all(state[k].shape == ours[k].shape for k in ours)


@needs_weights
def test_ported_weights_match_a_hand_rolled_forward_of_the_same_file():
    """Maps the weights through ``RRDBNet`` and through a conv-by-conv loop over the raw file.

    Proves the wiring (dense concatenations, both residual levels, nearest upsampling), not the
    paper.
    """
    raw = torch.load(PSNR_WEIGHTS, weights_only=True, map_location="cpu")

    def conv(x, name):
        return F.conv2d(x, raw[f"{name}.weight"], raw[f"{name}.bias"], padding=1)

    def act(x):
        return F.leaky_relu(x, 0.2)

    @torch.no_grad()
    def reference(x):
        fea = conv(x, "conv_first")
        h = fea
        for i in range(23):
            r = h
            for d in (1, 2, 3):
                p = f"RRDB_trunk.{i}.RDB{d}."
                x1 = act(conv(r, p + "conv1"))
                x2 = act(conv(torch.cat((r, x1), 1), p + "conv2"))
                x3 = act(conv(torch.cat((r, x1, x2), 1), p + "conv3"))
                x4 = act(conv(torch.cat((r, x1, x2, x3), 1), p + "conv4"))
                r = r + 0.2 * conv(torch.cat((r, x1, x2, x3, x4), 1), p + "conv5")
            h = h + 0.2 * r
        fea = fea + conv(h, "trunk_conv")
        for name in ("upconv1", "upconv2"):
            fea = act(conv(F.interpolate(fea, scale_factor=2, mode="nearest"), name))
        return conv(act(conv(fea, "HRconv")), "conv_last")

    net = RRDBNet()
    net.load_state_dict(load_official(PSNR_WEIGHTS))
    x = torch.rand(1, 3, 20, 24)

    with torch.no_grad():
        ours = net.eval()(x)

    torch.testing.assert_close(ours, reference(x), atol=1e-4, rtol=1e-4)


#: arXiv 1809.00219, Table 2, "RRDB (ours)", trained on DF2K, PSNR-Y x4. The released
#: RRDB_PSNR_x4.pth matches this row, not the DIV2K one (Set5 32.60 / Set14 28.88 / BSD100 27.76).
PAPER_DF2K = {"Set5_HR": 32.73, "Set14_HR": 28.99, "BSD100_HR": 27.85}


@needs_weights
@pytest.mark.parametrize("name", list(PAPER_DF2K))
def test_ported_psnr_weights_reproduce_the_papers_df2k_row(name):
    """x4 PSNR-Y within 0.05 dB: studio-range Y of the RGB output, border 4, no output rounding.

    The paper does not publish its evaluation protocol; this is the field's (and the
    authors' released code's) convention -- a single-channel image such as Set14's ``bridge``
    is scored as RGB, not on its raw plane (that would put Set14 0.087 dB below the paper).
    """
    if not (DATA / name).exists():
        pytest.skip(f"{name} not available")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = RRDBNet()
    net.load_state_dict(load_official(PSNR_WEIGHTS))
    net = net.to(device).eval()
    ds = ValidationDataset(DATA / name, 4)
    scores = []
    for i in range(len(ds)):
        lr, hr = (t[None].to(device) for t in ds[i])
        with torch.no_grad():
            sr = net(lr).clamp(0, 1)
        sy, hy = (rgb_to_ycbcr_studio(t)[:, :1, 4:-4, 4:-4] * 255 for t in (sr, hr))
        scores.append(float(10 * torch.log10(255.0**2 / ((sy - hy) ** 2).mean())))

    assert sum(scores) / len(scores) == pytest.approx(PAPER_DF2K[name], abs=0.05)
