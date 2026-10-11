"""The official-weights port: layout (always) and parity with the authors' release (data-gated).

The gated tests need ``data/weights/drcn/mat/DRCN_sf{2,3,4}.mat`` and the authors'
``code/result`` PNGs (see MANIFEST.md there) plus ``data/Set5_HR``. They skip when absent;
``SISR_DATA_DIR`` points them at another checkout's ``data/`` (a worktree has none).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from sisr.models.drcn import DRCN
from sisr.models.drcn.port import load_matconvnet
from tests.models.matlab_protocol import prepare as _prepare
from tests.models.matlab_protocol import psnr as _psnr
from tests.models.matlab_protocol import u8 as _u8
from tests.models.matlab_protocol import ycbcr2rgb as _ycbcr2rgb

scipy_io = pytest.importorskip("scipy.io")

DATA = Path(os.environ.get("SISR_DATA_DIR") or Path(__file__).resolve().parents[3] / "data")
WEIGHTS = DATA / "weights" / "drcn"
SET5 = DATA / "Set5_HR"
needs_weights = pytest.mark.skipif(
    not (WEIGHTS / "mat" / "DRCN_sf2.mat").exists(), reason="official DRCN weights not downloaded"
)
needs_authors_output = pytest.mark.skipif(
    not (WEIGHTS / "code" / "result" / "Set5" / "DRCN_x2" / "head_GT.png").exists()
    or not SET5.exists(),
    reason="authors' reference output or Set5 not available",
)


def _write_mat(path, f=4, d=2, seed=0):
    """A tiny file in the official layout: (H, W, Cin, Cout), singleton dims squeezed."""
    rng = np.random.default_rng(seed)
    shapes = {
        "filters1": (3, 3, f), "biases1": (f,),
        "filters2": (3, 3, f, f), "biases2": (f,),
        "filters19": (3, 3, f, f), "biases19": (f,),
        "filters20": (3, 3, f), "biases20": (),
        "filters_share": (3, 3, f, f), "biases_share": (f,),
        "ensemble_weight": (d + 1,),
    }  # fmt: skip
    values = {k: rng.standard_normal(s).astype(np.float32) for k, s in shapes.items()}
    params = np.array(
        [(k, v) for k, v in values.items()], dtype=[("name", object), ("value", object)]
    ).view(np.recarray)
    scipy_io.savemat(path, {"net": {"params": params}})
    return values


def test_load_matconvnet_transposes_every_filter_to_pytorch_layout(tmp_path):
    values = _write_mat(tmp_path / "t.mat")

    state = load_matconvnet(tmp_path / "t.mat")

    model = DRCN(in_out_channels=1, features=4, recursions=2)
    model.load_state_dict(state)  # keys and shapes line up
    np.testing.assert_array_equal(
        state["infer.weight"].numpy(), values["filters_share"].transpose(3, 2, 0, 1)
    )
    np.testing.assert_array_equal(
        state["embed.0.weight"].numpy()[:, 0], values["filters1"].transpose(2, 0, 1)
    )
    np.testing.assert_array_equal(
        state["recon2.weight"].numpy()[0], values["filters20"].transpose(2, 0, 1)
    )
    assert state["recon2.bias"].shape == (1,)
    np.testing.assert_array_equal(state["ensemble"].numpy(), values["ensemble_weight"])


def test_load_matconvnet_names_the_missing_parameter(tmp_path):
    _write_mat(tmp_path / "t.mat")
    data = scipy_io.loadmat(tmp_path / "t.mat", squeeze_me=True, struct_as_record=False)
    kept = [p for p in data["net"].params if p.name != "ensemble_weight"]
    params = np.array(
        [(p.name, p.value) for p in kept], dtype=[("name", object), ("value", object)]
    ).view(np.recarray)
    scipy_io.savemat(tmp_path / "bad.mat", {"net": {"params": params}})

    with pytest.raises(ValueError, match="ensemble_weight"):
        load_matconvnet(tmp_path / "bad.mat")


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _drcn(sf):
    model = DRCN(in_out_channels=1, features=256, recursions=16)
    model.load_state_dict(load_matconvnet(WEIGHTS / "mat" / f"DRCN_sf{sf}.mat"))
    return model.to(_device()).eval()


@torch.no_grad()
def _predict(model, y):
    x = torch.from_numpy(y).float()[None, None].to(_device())
    return model(x).primary.squeeze().double().cpu().numpy()


@needs_weights
def test_ported_weights_match_a_hand_rolled_forward_of_the_same_mat():
    """Maps the weights through ``DRCN`` and through a DagNN forward written from the .mat alone.

    Proves the mapping (transposes, shared convs, skip placement), not the paper.
    """
    net = scipy_io.loadmat(
        WEIGHTS / "mat" / "DRCN_sf2.mat", squeeze_me=True, struct_as_record=False
    )["net"]
    p = {q.name: torch.from_numpy(np.asarray(q.value, dtype=np.float32)) for q in net.params}

    def conv(x, name, bias):
        w = p[name]
        if w.ndim == 3:  # squeezed singleton: C_in for filters1, C_out for filters20
            w = w[:, :, None, :] if name == "filters1" else w[..., None]
        return F.conv2d(x, w.permute(3, 2, 0, 1), p[bias].reshape(-1), padding=1)

    @torch.no_grad()
    def reference(x):
        h = torch.relu(conv(torch.relu(conv(x, "filters1", "biases1")), "filters2", "biases2"))
        z = conv(h, "filters19", "biases19")
        preds = [conv(z, "filters20", "biases20")]
        for _ in range(16):
            z = conv(torch.relu(conv(z, "filters_share", "biases_share")), "filters19", "biases19")
            preds.append(conv(z, "filters20", "biases20"))
        return x + sum(w * q for w, q in zip(p["ensemble_weight"], preds, strict=True))

    x = torch.from_numpy(np.random.default_rng(0).uniform(0.06, 0.92, (1, 1, 24, 24))).float()
    model = DRCN(in_out_channels=1, features=256, recursions=16)
    model.load_state_dict(load_matconvnet(WEIGHTS / "mat" / "DRCN_sf2.mat"))

    with torch.no_grad():
        ours = model.eval()(x).primary

    torch.testing.assert_close(ours, reference(x), atol=1e-4, rtol=1e-4)


@needs_weights
@needs_authors_output
def test_ported_x2_reproduces_the_authors_released_output_image():
    """DRCN.m's own PNG for Set5 'head' (x2): within 2 grey levels everywhere."""
    _, y, cb, cr = _prepare(SET5 / "head.png", 2)

    pred = _predict(_drcn(2), y)  # primary already carries the input skip
    ours = _ycbcr2rgb(_u8(np.stack([pred, cb, cr], -1)))

    released = np.asarray(
        Image.open(WEIGHTS / "code" / "result" / "Set5" / "DRCN_x2" / "head_GT.png").convert("RGB")
    )
    assert ours.shape == released.shape
    diff = np.abs(ours.astype(int) - released.astype(int))
    assert diff.max() <= 2
    assert (diff > 1).mean() < 1e-3


PAPER_PSNR = {  # arXiv 1511.04491, benchmark table, PSNR-Y
    "Set5_HR": {2: 37.63, 3: 33.82, 4: 31.53},
    "Set14_HR": {2: 33.04, 3: 29.76, 4: 28.02},
    "BSD100_HR": {2: 31.85, 3: 28.80, 4: 27.23},
}


@needs_weights
@pytest.mark.parametrize("sf", [2, 3, 4])
@pytest.mark.parametrize("name", list(PAPER_PSNR))
def test_ported_weights_reproduce_the_papers_psnr(name, sf):
    """PSNR-Y under the authors' protocol within 0.05 dB of the paper (issue #296's rule).

    Set14's single-channel ``bridge`` pins the grey branch: forced through RGB it scores
    1.32 dB higher and lifts the Set14 mean by ~0.09 dB.
    """
    if not (DATA / name).exists():
        pytest.skip(f"{name} not available")
    model = _drcn(sf)
    scores = []
    for path in sorted((DATA / name).glob("*.png")):
        gt, y, _, _ = _prepare(path, sf)
        scores.append(_psnr(gt, _u8(_predict(model, y)), sf))

    assert np.mean(scores) == pytest.approx(PAPER_PSNR[name][sf], abs=0.05)
