"""Tests for sisr.export.to_onnx (INIT.7).

Skips cleanly when the optional `onnx` / `onnxruntime` extra is absent, so the
suite stays green for contributors who don't install `.[export]`. The CI
`onnx-export` job in `.github/workflows/test.yml` installs the extra so these
tests actually run somewhere.
"""

import sys

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

from sisr.export import to_onnx  # noqa: E402
from sisr.models.srcnn import SRCNN, SRCNNTrainingConfig  # noqa: E402
from sisr.models.srresnet import SRResNet, SRResNetTrainingConfig  # noqa: E402
from sisr.processors import RGBProcessor, YChannelProcessor  # noqa: E402
from sisr.training import SRLightning  # noqa: E402


def _make_srcnn_module(example_input_shape: tuple[int, ...] | None) -> SRLightning:
    """Tiny Y-channel SRCNN — small enough to trace/export quickly."""
    model = SRCNN(num_channels=1, num_filters=(8, 4), kernel_sizes=(5, 1, 3), padding=0)
    training_config = SRCNNTrainingConfig(example_input_shape=example_input_shape)
    return SRLightning(model=model, processor=YChannelProcessor(), training_config=training_config)


def _make_srresnet_module(example_input_shape: tuple[int, ...] | None) -> SRLightning:
    """Tiny RGB SRResNet (x4) — small enough to trace/export quickly."""
    model = SRResNet(
        scale=4,
        in_out_channels=3,
        hidden_channel=8,
        kernel_sizes=(9, 3, 9),
        num_residual_blocks=2,
        padding="same",
    )
    training_config = SRResNetTrainingConfig(example_input_shape=example_input_shape)
    return SRLightning(model=model, processor=RGBProcessor(), training_config=training_config)


def _run_ort(onnx_path, x: torch.Tensor) -> np.ndarray:
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (out,) = session.run(None, {"input": x.numpy()})
    return out


@pytest.mark.parametrize(
    ("make_module", "export_shape", "sizes"),
    [
        (_make_srcnn_module, (1, 48, 48), [(48, 48), (30, 70)]),
        (_make_srresnet_module, (3, 16, 16), [(16, 16), (20, 28)]),
    ],
    ids=["srcnn", "srresnet"],
)
def test_to_onnx_parity_and_dynamic_axes(tmp_path, make_module, export_shape, sizes):
    """Exported graph matches torch eager at the export size AND a different one.

    The second (different) size in `sizes` is the load-bearing assertion for
    `dynamic_axes`: a fixed-shape export would only pass the first.
    """
    module = make_module(export_shape)
    onnx_path = tmp_path / "model.onnx"

    to_onnx(module, onnx_path)

    model = module.model
    model.eval()
    channels = export_shape[0]
    for h, w in sizes:
        x = torch.rand(1, channels, h, w)
        with torch.no_grad():
            expected = model(x).numpy()
        actual = _run_ort(onnx_path, x)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)


def test_to_onnx_requires_example_input_shape_or_input_sample():
    """No input_sample and no training_config.example_input_shape -> ValueError."""
    model = SRCNN(num_channels=1, num_filters=(8, 4), kernel_sizes=(5, 1, 3), padding=0)
    module = SRLightning(model=model, processor=YChannelProcessor())

    with pytest.raises(ValueError, match="example_input_shape"):
        to_onnx(module, "unused.onnx")


def test_to_onnx_accepts_explicit_input_sample(tmp_path):
    """An explicit input_sample bypasses the training_config.example_input_shape seam."""
    model = SRCNN(num_channels=1, num_filters=(8, 4), kernel_sizes=(5, 1, 3), padding=0)
    module = SRLightning(model=model, processor=YChannelProcessor())
    onnx_path = tmp_path / "model.onnx"

    to_onnx(module, onnx_path, input_sample=torch.zeros(1, 1, 24, 24))

    model.eval()
    x = torch.rand(1, 1, 40, 40)
    with torch.no_grad():
        expected = model(x).numpy()
    actual = _run_ort(onnx_path, x)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)


def test_to_onnx_ckpt_path_loads_weights_before_export(tmp_path):
    """ckpt_path's state_dict is loaded into the module before tracing."""
    module = _make_srcnn_module(example_input_shape=(1, 32, 32))
    trained = _make_srcnn_module(example_input_shape=(1, 32, 32))
    with torch.no_grad():
        for p in trained.model.parameters():
            p.add_(1.0)
    assert not torch.equal(module.model.feat[0].weight, trained.model.feat[0].weight)

    ckpt_path = tmp_path / "trained.ckpt"
    torch.save({"state_dict": trained.state_dict()}, ckpt_path)

    onnx_path = tmp_path / "model.onnx"
    to_onnx(module, onnx_path, ckpt_path=ckpt_path)

    # The live module itself now carries the checkpoint's weights...
    assert torch.equal(module.model.feat[0].weight, trained.model.feat[0].weight)

    # ...and so does the exported graph.
    trained.model.eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        expected = trained.model(x).numpy()
    actual = _run_ort(onnx_path, x)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)


def test_to_onnx_restores_training_mode(tmp_path):
    """model.training is restored after export, whatever it was before."""
    module = _make_srcnn_module(example_input_shape=(1, 16, 16))
    module.model.train()

    to_onnx(module, tmp_path / "model.onnx")

    assert module.model.training is True


def test_to_onnx_missing_onnx_raises_import_error_with_extra_hint(monkeypatch):
    """Simulates `onnx` missing (sys.modules poisoning) -> ImportError mentioning the extra."""
    monkeypatch.setitem(sys.modules, "onnx", None)
    module = _make_srcnn_module(example_input_shape=(1, 16, 16))

    with pytest.raises(ImportError, match=r"\[export\]"):
        to_onnx(module, "unused.onnx")
