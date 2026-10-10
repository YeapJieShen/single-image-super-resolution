"""The official-weights port: layout (always) and parity with the authors' release (data-gated).

The gated tests need ``data/weights/vdsr/mat/VDSR.mat`` and the authors' ``code/`` directory
(see MANIFEST.md there) plus ``data/Set5_HR`` and friends. They skip when absent;
``SISR_DATA_DIR`` points them at another checkout's ``data/`` (a worktree has none).
"""

import io
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from sisr.models.vdsr import VDSR
from sisr.models.vdsr.port import load_matconvnet, mcos_single_arrays, state_dict_from_arrays
from tests.models.matlab_protocol import prepare, score_image, u8, ycbcr2rgb

scipy_io = pytest.importorskip("scipy.io")

DATA = Path(os.environ.get("SISR_DATA_DIR") or Path(__file__).resolve().parents[3] / "data")
WEIGHTS = DATA / "weights" / "vdsr"
needs_weights = pytest.mark.skipif(
    not (WEIGHTS / "mat" / "VDSR.mat").exists(), reason="official VDSR weights not downloaded"
)
needs_authors_output = pytest.mark.skipif(
    not (WEIGHTS / "code" / "result" / "Set5" / "VDSR_x2" / "baby_GT.png").exists()
    or not (DATA / "Set5_HR").exists(),
    reason="authors' reference output or Set5 not available",
)


def _workspace(arrays: list[np.ndarray]) -> bytes:
    """A MAT-file element stream shaped like a GPU-saved net's function workspace.

    Each array sits in its own slot of a cell, next to a float64 scalar, as the real
    file's ``gpuArray`` objects do; ``scipy`` can write that, though not the opaque object.
    """
    cell = np.empty((len(arrays), 1), dtype=object)
    for i, a in enumerate(arrays):
        slot = np.empty((1, 2), dtype=object)
        slot[0, 0], slot[0, 1] = np.float64(i), a
        cell[i, 0] = slot
    buf = io.BytesIO()
    scipy_io.savemat(buf, {"mcos": cell})
    return buf.getvalue()[128:]  # the stream after the 128-byte file header


def _layers(depth=3, f=4, seed=0):
    """Per layer: filters (H, W, Cin, Cout), biases, and two momentum buffers (to be skipped)."""
    rng = np.random.default_rng(seed)
    widths = [1] + [f] * (depth - 1) + [1]
    out, filters = [], []
    for i in range(depth):
        w = rng.standard_normal((3, 3, widths[i], widths[i + 1])).astype(np.float32)
        if widths[i + 1] == 1:
            w = w[..., 0]  # MATLAB drops a trailing singleton dimension on save
        b = rng.standard_normal((1, widths[i + 1])).astype(np.float32)
        junk = rng.standard_normal((3, 3, widths[i], widths[i + 1])).astype(np.float32)
        out += [w, b, junk, b * 0 + 7]
        filters.append((w, b))
    return out, filters


def test_the_workspace_walker_returns_float32_arrays_with_matlab_dims_in_order():
    arrays, _ = _layers()

    found = mcos_single_arrays(_workspace(arrays))

    assert len(found) == len(arrays)
    for a, (dims, flat) in zip(arrays, found, strict=True):
        assert tuple(dims) == a.shape
        np.testing.assert_array_equal(flat.reshape(dims, order="F"), a)


def test_state_dict_transposes_every_filter_to_pytorch_layout():
    arrays, filters = _layers()

    state = state_dict_from_arrays(mcos_single_arrays(_workspace(arrays)))

    model = VDSR(depth=3, features=4)
    model.load_state_dict(state)  # keys and shapes line up
    for i, (w, b) in enumerate(filters):
        w4 = w if w.ndim == 4 else w[..., None]  # the last layer has one output channel
        np.testing.assert_array_equal(
            state[f"body.{2 * i}.weight"].numpy(), w4.transpose(3, 2, 0, 1)
        )
        np.testing.assert_array_equal(state[f"body.{2 * i}.bias"].numpy(), b.reshape(-1))


def test_state_dict_rejects_arrays_that_are_not_four_per_layer():
    arrays, _ = _layers()

    with pytest.raises(ValueError, match="4 float32 arrays per layer"):
        state_dict_from_arrays(mcos_single_arrays(_workspace(arrays[:-1])))


def test_load_matconvnet_names_a_file_with_no_function_workspace(tmp_path):
    scipy_io.savemat(tmp_path / "plain.mat", {"x": np.zeros(3)})

    with pytest.raises(ValueError, match="function_workspace"):
        load_matconvnet(tmp_path / "plain.mat")


# --- against the authors' release ------------------------------------------------------


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _vdsr():
    model = VDSR()
    model.load_state_dict(load_matconvnet(WEIGHTS / "mat" / "VDSR.mat"))
    return model.to(_device()).eval()


def _predictor(model):
    @torch.no_grad()
    def predict(y):
        x = torch.from_numpy(y).float()[None, None].to(_device())
        return model(x).squeeze().double().cpu().numpy()

    return predict


@needs_weights
def test_ported_weights_match_a_hand_rolled_forward_of_the_same_mat():
    """Maps the weights through ``VDSR`` and through a plain conv/ReLU loop with the global skip.

    Proves the wiring (transposes, skip placement), not the paper.
    """
    state = load_matconvnet(WEIGHTS / "mat" / "VDSR.mat")
    x = torch.from_numpy(np.random.default_rng(0).uniform(0.06, 0.92, (1, 1, 24, 24))).float()

    @torch.no_grad()
    def reference(x):
        h = x
        for i in range(20):
            h = F.conv2d(h, state[f"body.{2 * i}.weight"], state[f"body.{2 * i}.bias"], padding=1)
            if i < 19:
                h = torch.relu(h)
        return x + h

    with torch.no_grad():
        ours = _vdsr().cpu()(x)

    torch.testing.assert_close(ours, reference(x), atol=1e-4, rtol=1e-4)


@needs_weights
@needs_authors_output
@pytest.mark.parametrize("name", ["baby", "bird", "butterfly", "head", "woman"])
def test_ported_x2_reproduces_the_authors_released_output_and_printed_psnr(name):
    """VDSR.m's own PNG for Set5 (x2) within 2 grey levels, and the PSNR it printed."""
    predict = _predictor(_vdsr())
    path = DATA / "Set5_HR" / f"{name}.png"
    _, y, cb, cr = prepare(path, 2)

    ours = ycbcr2rgb(u8(np.stack([predict(y), cb, cr], -1)))

    released_dir = WEIGHTS / "code" / "result" / "Set5" / "VDSR_x2"
    released = np.asarray(Image.open(released_dir / f"{name}_GT.png").convert("RGB"))
    assert ours.shape == released.shape
    assert np.abs(ours.astype(int) - released.astype(int)).max() <= 2
    table = scipy_io.loadmat(str(released_dir / "psnr_ssim.mat"))["evaltable"][:, 0]
    # evaltable rows follow the sorted directory listing ('.' and '..' first).
    row = 2 + ["baby", "bird", "butterfly", "head", "woman"].index(name)
    assert score_image(path, 2, predict) == pytest.approx(
        table[row], abs=5e-3
    )  # CPU is within 2e-4; cudnn's TF32 convs, 1e-3


PAPER_PSNR = {  # arXiv 1511.04587, Table 3, VDSR, PSNR-Y
    "Set5_HR": {2: 37.53, 3: 33.66, 4: 31.35},
    "Set14_HR": {2: 33.03, 3: 29.77, 4: 28.01},
    "BSD100_HR": {2: 31.90, 3: 28.82, 4: 27.29},
}


@needs_weights
@pytest.mark.parametrize("sf", [2, 3, 4])
@pytest.mark.parametrize("name", list(PAPER_PSNR))
def test_ported_weights_reproduce_the_papers_psnr(name, sf):
    """One model for all three scales, scored under the authors' protocol, within 0.05 dB.

    Set14's single-channel ``bridge`` pins the grey branch: forced through RGB it scores
    1.32 dB higher and lifts the Set14 mean by ~0.09 dB.
    """
    if not (DATA / name).exists():
        pytest.skip(f"{name} not available")
    predict = _predictor(_vdsr())
    scores = [score_image(p, sf, predict) for p in sorted((DATA / name).glob("*.png"))]

    assert np.mean(scores) == pytest.approx(PAPER_PSNR[name][sf], abs=0.05)


# --- through the pipeline, with the studio-range processor -----------------------------


def _pipeline_psnr(name, sf, processor):
    from sisr.datasets.pre_upsampled import ValidationDataset
    from sisr.models.vdsr import VDSREvalConfig, VDSRTrainingConfig
    from sisr.training import SRLightning
    from tests.models.matlab_protocol import psnr_float_y

    model = VDSR()
    lit = SRLightning(
        model=model,
        processor=processor,
        training_config=VDSRTrainingConfig(scale=[sf]),
        eval_config=VDSREvalConfig(),
    )
    model.load_state_dict(load_matconvnet(WEIGHTS / "mat" / "VDSR.mat"))  # after construction
    lit = lit.to(_device()).eval()
    ds = ValidationDataset(DATA / name, sf)
    scores = []
    for i in range(len(ds)):
        lr, hr = (t[None].to(_device()) for t in ds[i])
        with torch.no_grad():
            sr, hr_c = lit.predict_rgb(lr, hr)
        scores.append(psnr_float_y(sr, hr_c, sf))
    return float(np.mean(scores))


@needs_weights
def test_studio_y_processor_brings_vdsr_set5_x2_to_the_paper_through_the_pipeline():
    """The released weights were trained on MATLAB's studio Y, not the full-range one.

    Set5 x2, float-Y metric (no 8-bit rounding): full-range 37.31, studio 37.50, paper 37.53.
    """
    from sisr.processors import StudioYChannelProcessor, YChannelProcessor

    if not (DATA / "Set5_HR").exists():
        pytest.skip("Set5_HR not available")
    studio = _pipeline_psnr("Set5_HR", 2, StudioYChannelProcessor())
    full = _pipeline_psnr("Set5_HR", 2, YChannelProcessor())

    assert studio == pytest.approx(37.53, abs=0.05)
    assert abs(studio - 37.53) < abs(full - 37.53) - 0.1
