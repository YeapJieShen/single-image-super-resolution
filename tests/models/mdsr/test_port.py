"""The official-weights port: names (always) and parity with the authors' release (data-gated).

The gated tests need ``data/weights/mdsr/*.pt`` (see MANIFEST.md there) and the benchmark sets.
They skip when absent; ``SISR_DATA_DIR`` points them at another checkout's ``data/``.
"""

import os
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from sisr.datasets.native_lr import ValidationDataset
from sisr.models.mdsr import MDSR, MDSREvalConfig, MDSRTrainingConfig
from sisr.models.mdsr.port import load_official, rename_key
from sisr.processors import RGBProcessor
from sisr.training import SRLightning
from tests.models.matlab_protocol import psnr_float_y, psnr_matlab, psnr_matlab_eval

DATA = Path(os.environ.get("SISR_DATA_DIR") or Path(__file__).resolve().parents[3] / "data")
WEIGHTS = DATA / "weights" / "mdsr"
BASELINE = WEIGHTS / "mdsr_baseline-a00cab12.pt"
FULL = WEIGHTS / "mdsr-4a78bedf.pt"
needs_weights = pytest.mark.skipif(
    not (BASELINE.exists() and FULL.exists()), reason="official MDSR weights absent"
)


@pytest.mark.parametrize(
    "official, ours",
    [
        ("head.0.weight", "head.weight"),
        ("pre_process.1.0.body.0.weight", "pre_process.1.0.conv1.weight"),
        ("pre_process.2.1.body.2.bias", "pre_process.2.1.conv2.bias"),
        ("body.5.body.2.bias", "blocks.5.conv2.bias"),
        ("body.16.weight", "body_conv.weight"),
        ("upsample.0.0.weight", "up.0.0.weight"),
        ("upsample.2.2.bias", "up.2.1.bias"),
        ("tail.0.weight", "tail.weight"),
    ],
)
def test_official_names_map_to_ours(official, ours):
    assert rename_key(official, num_blocks=16) == ours


@pytest.mark.parametrize("bad", ["body.3.weight", "upsample.0.1.weight.x", "tail.1.weight"])
def test_an_unknown_official_name_is_refused(bad):
    with pytest.raises(ValueError, match="unexpected"):
        rename_key(bad, num_blocks=16)


def _official_state(net: MDSR) -> dict[str, torch.Tensor]:
    n = net.hparams["num_blocks"]
    out = {}
    for k, v in net.state_dict().items():
        k = re.sub(
            r"^pre_process\.(\d+)\.(\d+)\.conv([12])\.",
            lambda m: f"pre_process.{m[1]}.{m[2]}.body.{(int(m[3]) - 1) * 2}.",
            k,
        )
        k = re.sub(
            r"^blocks\.(\d+)\.conv([12])\.", lambda m: f"body.{m[1]}.body.{(int(m[2]) - 1) * 2}.", k
        )
        k = (
            k.replace("body_conv.", f"body.{n}.")
            .replace("head.", "head.0.")
            .replace("tail.", "tail.0.")
        )
        k = re.sub(r"^up\.(\d+)\.(\d+)\.", lambda m: f"upsample.{m[1]}.{int(m[2]) * 2}.", k)
        out[k] = v
    mean = net.mean.flatten() * net.rgb_range
    eye = torch.eye(3).view(3, 3, 1, 1)
    out.update({"sub_mean.weight": eye, "sub_mean.bias": -mean, "add_mean.weight": eye})
    out["add_mean.bias"] = mean
    return out


def test_a_renamed_state_dict_loads_strictly_and_drops_the_mean_shift(tmp_path):
    net = MDSR(features=8, num_blocks=2)
    torch.save(_official_state(net), tmp_path / "x.pt")

    state = load_official(tmp_path / "x.pt", MDSR(features=8, num_blocks=2))

    assert state.keys() == net.state_dict().keys()
    MDSR(features=8, num_blocks=2).load_state_dict(state, strict=True)


def test_a_checkpoint_with_a_different_mean_is_refused(tmp_path):
    net = MDSR(features=8, num_blocks=2)
    state = _official_state(net)
    state["add_mean.bias"] = state["add_mean.bias"] + 1.0
    torch.save(state, tmp_path / "x.pt")

    with pytest.raises(ValueError, match="add_mean"):
        load_official(tmp_path / "x.pt", net)


@needs_weights
@pytest.mark.parametrize("path, blocks", [(BASELINE, 16), (FULL, 80)], ids=["baseline", "full"])
def test_both_released_checkpoints_load_strictly(path, blocks):
    net = MDSR(num_blocks=blocks)

    net.load_state_dict(load_official(path, net), strict=True)


@needs_weights
@pytest.mark.parametrize("scale", [2, 3, 4])
def test_ported_weights_match_a_hand_rolled_forward_of_the_same_file(scale):
    """Maps the weights through ``MDSR`` and a conv-by-conv loop over the raw file written from the
    authors' ``mdsr.py``: proves the per-scale routing and wiring, not the paper."""
    raw = torch.load(BASELINE, weights_only=True, map_location="cpu")
    idx = (2, 3, 4).index(scale)

    def conv(x, name, k=3):
        return F.conv2d(x, raw[f"{name}.weight"], raw[f"{name}.bias"], padding=k // 2)

    def resblock(x, p, k):
        return x + conv(torch.relu(conv(x, f"{p}.body.0", k)), f"{p}.body.2", k)

    @torch.no_grad()
    def reference(x01):
        x = conv(x01 * 255.0 + raw["sub_mean.bias"].view(1, 3, 1, 1), "head.0")
        for j in (0, 1):
            x = resblock(x, f"pre_process.{idx}.{j}", 5)
        res = x
        for i in range(16):
            res = resblock(res, f"body.{i}", 3)
        res = conv(res, "body.16") + x
        for k in [0, 2] if scale == 4 else [0]:
            res = F.pixel_shuffle(conv(res, f"upsample.{idx}.{k}"), 3 if scale == 3 else 2)
        return (conv(res, "tail.0") + raw["add_mean.bias"].view(1, 3, 1, 1)) / 255.0

    net = MDSR(num_blocks=16)
    net.load_state_dict(load_official(BASELINE, net))
    x = torch.rand(1, 3, 10, 12)

    with torch.no_grad():
        ours = net.eval()(x, scale=scale)

    torch.testing.assert_close(ours, reference(x), atol=1e-4, rtol=1e-4)


#: arXiv 1707.02921, Table 3, MDSR and MDSR+ (the 80-block model), PSNR-Y.
PAPER = {
    "Set5_HR": {2: (38.11, 38.17), 3: (34.66, 34.77), 4: (32.50, 32.60)},
    "Set14_HR": {2: (33.85, 33.92), 3: (30.44, 30.53), 4: (28.72, 28.82)},
    "BSD100_HR": {2: (32.29, 32.34), 3: (29.25, 29.30), 4: (27.72, 27.78)},
}


def _scores(name, scale, plus, metrics):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MDSR(num_blocks=80)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=MDSRTrainingConfig(),
        eval_config=MDSREvalConfig(),
    )
    model.load_state_dict(load_official(FULL, model))  # after construction
    lit = lit.to(device).eval()
    ds = ValidationDataset(DATA / name, scale)
    scores = {m.__name__: [] for m in metrics}
    for i in range(len(ds)):
        lr, hr = (t[None].to(device) for t in ds[i])
        with torch.no_grad():
            sr, hr_cropped = lit.predict_rgb(lr, hr, self_ensemble=plus)
        with Image.open(ds.img_paths[i]) as img:
            grey = img.mode == "L"
        for m in metrics:
            extra = (grey,) if m is psnr_matlab_eval else ()
            scores[m.__name__].append(m(sr, hr_cropped, scale, *extra))
    return {k: sum(v) / len(v) for k, v in scores.items()}


@needs_weights
@pytest.mark.parametrize("scale", [2, 3, 4])
@pytest.mark.parametrize("name", list(PAPER))
@pytest.mark.parametrize("plus", [False, True], ids=["MDSR", "MDSR+"])
def test_released_mdsr_is_within_0_2_db_of_the_papers_numbers(name, scale, plus):
    """The released 80-block checkpoint under the paper's protocol (MATLAB 8-bit luma).

    Within 0.2 dB in every cell (the pre-registered limit), not a 0.05 dB reproduction; 11 of 18
    cells are within 0.05 dB. Two differences are named. Set14: NTIRE2017's ``demo/evaluation.m``
    (https://github.com/LimBee/NTIRE2017) scores a single-channel target on its raw plane, and
    Set14's ``bridge`` is one; ``psnr_matlab_eval`` does the same and moves the Set14 mean by
    -0.094 dB. Set5: the authors' PyTorch release is a different network from the Torch7 one the
    paper tabulated (https://cv.snu.ac.kr/research/EDSR/model_paper.tar has no scale-specific
    pre-processing and has per-scale tails), and its own README's DIV2K table sits below the
    paper's. The float-Y figure reads 0.01-0.10 dB higher.
    """
    if not (DATA / name).exists():
        pytest.skip(f"{name} not available")

    got = _scores(name, scale, plus, [psnr_matlab, psnr_matlab_eval, psnr_float_y])

    paper = PAPER[name][scale][int(plus)]
    assert got["psnr_matlab"] == pytest.approx(paper, abs=0.2)
    assert got["psnr_matlab_eval"] == pytest.approx(paper, abs=0.2)
    assert got["psnr_float_y"] == pytest.approx(paper, abs=0.25)
