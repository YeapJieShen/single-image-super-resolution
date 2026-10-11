"""The official-weights port: names and mean shift (always) and parity (data-gated).

The gated tests need ``data/weights/edsr/*.pt`` (see MANIFEST.md there) and the benchmark sets.
They skip when absent; ``SISR_DATA_DIR`` points them at another checkout's ``data/`` (a worktree
has none).
"""

import os
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sisr.datasets.native_lr import ValidationDataset
from sisr.models.edsr import EDSR, EDSREvalConfig, EDSRTrainingConfig
from sisr.models.edsr.port import load_official, rename_key
from sisr.processors import RGBProcessor
from sisr.training import SRLightning
from tests.models.matlab_protocol import psnr_float_y, psnr_matlab

DATA = Path(os.environ.get("SISR_DATA_DIR") or Path(__file__).resolve().parents[3] / "data")
WEIGHTS = DATA / "weights" / "edsr"
FULL = {2: "edsr_x2-0edfb8a3.pt", 3: "edsr_x3-ea3ef2c6.pt", 4: "edsr_x4-4f62e9ef.pt"}
BASELINE = {
    2: "edsr_baseline_x2-1bc95232.pt",
    3: "edsr_baseline_x3-abf2a44e.pt",
    4: "edsr_baseline_x4-6b446fab.pt",
}
needs_weights = pytest.mark.skipif(
    not all((WEIGHTS / f).exists() for f in [*FULL.values(), *BASELINE.values()]),
    reason="official EDSR weights absent",
)


@pytest.mark.parametrize(
    "official, ours",
    [
        ("head.0.weight", "head.weight"),
        ("body.3.body.0.bias", "blocks.3.conv1.bias"),
        ("body.15.body.2.weight", "blocks.15.conv2.weight"),
        ("body.16.weight", "body_conv.weight"),
        ("tail.0.0.weight", "up.0.weight"),
        ("tail.0.2.bias", "up.1.bias"),
        ("tail.1.weight", "tail.weight"),
    ],
)
def test_official_names_map_to_ours(official, ours):
    assert rename_key(official, num_blocks=16) == ours


def test_an_unknown_official_name_is_refused():
    with pytest.raises(ValueError, match="unexpected"):
        rename_key("body.3.body.1.weight", num_blocks=16)  # the ReLU has no parameters


def _official_state(net: EDSR) -> dict[str, torch.Tensor]:
    """Our state dict under the authors' names, plus their mean-shift convs."""
    n = net.hparams["num_blocks"]
    out = {}
    for k, v in net.state_dict().items():
        k = re.sub(
            r"^blocks\.(\d+)\.conv([12])\.", lambda m: f"body.{m[1]}.body.{(int(m[2]) - 1) * 2}.", k
        )
        k = (
            k.replace("body_conv.", f"body.{n}.")
            .replace("head.", "head.0.")
            .replace("tail.", "tail.1.")
        )
        k = re.sub(r"^up\.(\d+)\.", lambda m: f"tail.0.{int(m[1]) * 2}.", k)
        out[k] = v
    mean = net.mean.flatten() * net.rgb_range
    eye = torch.eye(3).view(3, 3, 1, 1)
    out.update({"sub_mean.weight": eye, "sub_mean.bias": -mean, "add_mean.weight": eye})
    out["add_mean.bias"] = mean
    return out


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_a_renamed_state_dict_loads_strictly_and_drops_the_mean_shift(tmp_path, scale):
    net = EDSR(scale=scale, features=8, num_blocks=2)
    path = tmp_path / "x.pt"
    torch.save(_official_state(net), path)

    state = load_official(path, EDSR(scale=scale, features=8, num_blocks=2))

    assert state.keys() == net.state_dict().keys()
    EDSR(scale=scale, features=8, num_blocks=2).load_state_dict(state, strict=True)


def test_a_checkpoint_with_a_different_mean_is_refused(tmp_path):
    net = EDSR(scale=2, features=8, num_blocks=2)
    state = _official_state(net)
    state["sub_mean.bias"] = state["sub_mean.bias"] + 1.0
    torch.save(state, tmp_path / "x.pt")

    with pytest.raises(ValueError, match="sub_mean"):
        load_official(tmp_path / "x.pt", net)


@needs_weights
@pytest.mark.parametrize("scale", [2, 3, 4])
@pytest.mark.parametrize("preset", ["baseline", "full"])
def test_every_released_checkpoint_loads_strictly(preset, scale):
    full = preset == "full"
    net = EDSR(
        scale=scale,
        features=256 if full else 64,
        num_blocks=32 if full else 16,
        residual_scale=0.1 if full else 1.0,
    )
    file = (FULL if full else BASELINE)[scale]

    net.load_state_dict(load_official(WEIGHTS / file, net), strict=True)


@needs_weights
def test_ported_weights_match_a_hand_rolled_forward_of_the_same_file():
    """Maps the weights through ``EDSR`` and through a conv-by-conv loop over the raw file
    written from the authors' ``edsr.py`` (0-255 range, mean shift as an add). Proves the wiring
    (mean shift, residual blocks, pixel-shuffle layout), not the paper."""
    raw = torch.load(WEIGHTS / BASELINE[4], weights_only=True, map_location="cpu")

    def conv(x, name):
        return F.conv2d(x, raw[f"{name}.weight"], raw[f"{name}.bias"], padding=1)

    @torch.no_grad()
    def reference(x01):
        x = x01 * 255.0 + raw["sub_mean.bias"].view(1, 3, 1, 1)
        x = conv(x, "head.0")
        res = x
        for i in range(16):
            res = res + conv(torch.relu(conv(res, f"body.{i}.body.0")), f"body.{i}.body.2")
        res = conv(res, "body.16") + x
        for name in ("tail.0.0", "tail.0.2"):
            res = F.pixel_shuffle(conv(res, name), 2)
        return (conv(res, "tail.1") + raw["add_mean.bias"].view(1, 3, 1, 1)) / 255.0

    net = EDSR(scale=4)
    net.load_state_dict(load_official(WEIGHTS / BASELINE[4], net))
    x = torch.rand(1, 3, 12, 14)

    with torch.no_grad():
        ours = net.eval()(x)

    torch.testing.assert_close(ours, reference(x), atol=1e-4, rtol=1e-4)


#: arXiv 1707.02921, Table 3: EDSR and EDSR+ (the 32-block, 256-feature model), PSNR-Y.
PAPER = {
    "Set5_HR": {2: (38.11, 38.20), 3: (34.65, 34.76), 4: (32.46, 32.62)},
    "Set14_HR": {2: (33.92, 34.02), 3: (30.52, 30.66), 4: (28.80, 28.94)},
    "BSD100_HR": {2: (32.32, 32.37), 3: (29.25, 29.32), 4: (27.71, 27.79)},
}


def _scores(name, scale, plus, metrics):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EDSR(scale=scale, features=256, num_blocks=32, residual_scale=0.1)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=EDSRTrainingConfig(scale=scale),
        eval_config=EDSREvalConfig(),
    )
    model.load_state_dict(load_official(WEIGHTS / FULL[scale], model))  # after construction
    lit = lit.to(device).eval()
    ds = ValidationDataset(DATA / name, scale)
    scores = {m.__name__: [] for m in metrics}
    for i in range(len(ds)):
        lr, hr = (t[None].to(device) for t in ds[i])
        with torch.no_grad():
            sr, hr_cropped = lit.predict_rgb(lr, hr, self_ensemble=plus)
        for m in metrics:
            scores[m.__name__].append(m(sr, hr_cropped, scale))
    return {k: sum(v) / len(v) for k, v in scores.items()}


@needs_weights
@pytest.mark.parametrize("scale", [2, 3, 4])
@pytest.mark.parametrize("name", list(PAPER))
@pytest.mark.parametrize("plus", [False, True], ids=["EDSR", "EDSR+"])
def test_released_edsr_reproduces_the_papers_numbers_within_0_05_db(name, scale, plus):
    """The released checkpoints under the paper's own protocol: all 18 cells within 0.05 dB.

    The paper evaluated with MATLAB, whose ``rgb2ycbcr`` on 8-bit input returns 8-bit luma, so
    both planes are rounded (``psnr_matlab``). The pipeline's float Y (``psnr_float_y``) reads
    0.01-0.10 dB higher on the same outputs, so that figure is only bounded at 0.2 dB.
    """
    if not (DATA / name).exists():
        pytest.skip(f"{name} not available")

    got = _scores(name, scale, plus, [psnr_matlab, psnr_float_y])

    paper = PAPER[name][scale][int(plus)]
    assert got["psnr_matlab"] == pytest.approx(paper, abs=0.05)
    assert got["psnr_float_y"] == pytest.approx(paper, abs=0.2)
