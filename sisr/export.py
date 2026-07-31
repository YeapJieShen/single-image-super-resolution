"""Bare-model ONNX export (INIT.7).

Exports ``module.model`` — the wrapped :class:`~sisr.models.base.SRModel`,
*without* the :class:`~sisr.processors.SRProcessor` — so the graph matches
:meth:`SRLightning.forward` exactly (which is itself model-only; see
``sisr/training/lightning_module.py``). This is a deliberate choice, not an
oversight: a consumer of the exported graph is expected to have ``sisr``
importable and call ``processor.extract`` / ``processor.reconstruct``
themselves.

For :class:`~sisr.processors.RGBProcessor` architectures (SRResNet), those
methods are identity functions, so the bare graph is already an end-to-end
LR-RGB -> SR-RGB pipeline. For :class:`~sisr.processors.YChannelProcessor`
architectures (SRCNN), the graph only covers Y -> Y: reconstructing an RGB
image needs the LR image's Cb/Cr channels back (bicubic-upsampled to the SR
size), which this export omits. A non-Python ONNX consumer of an SRCNN graph
must reimplement that chroma path itself — see the README's ONNX section.

Requires the optional ``export`` extra (``onnx``, ``onnxruntime``). The
import is gated inside :func:`to_onnx` so plain ``import sisr`` — and
``import sisr.export`` itself — stays free of the dependency until export is
actually invoked.
"""

from __future__ import annotations

import warnings
from os import PathLike
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .training import SRLightning

_EXTRA_HINT = (
    "ONNX export requires the optional 'export' extra: "
    "pip install '.[export]' (or: python -m uv pip install '.[export]')."
)


def _require_onnx() -> None:
    """Raise a clear ``ImportError`` if ``onnx`` is not installed."""
    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise ImportError(f"onnx is not installed. {_EXTRA_HINT}") from e


def to_onnx(
    module: SRLightning,
    file_path: str | PathLike,
    *,
    ckpt_path: str | PathLike | None = None,
    input_sample: torch.Tensor | None = None,
    opset_version: int = 17,
) -> None:
    """Export ``module.model`` (the bare SR model) to ONNX with dynamic H/W.

    Traces the wrapped model alone — see the module docstring for why the
    processor is excluded and what that means for SRCNN consumers. The
    exported graph accepts arbitrary spatial dimensions (``dynamic_axes`` on
    height/width): a fixed-shape export would be useless on real images,
    since ``training_config.example_input_shape`` is only a TensorBoard-graph
    dummy size, not a training or inference constraint.

    Args:
        module: An :class:`~sisr.training.SRLightning` instance (e.g. built
            from YAML by :class:`~sisr.cli.SRLightningCLI`). Only
            ``module.model`` is traced.
        file_path: Destination ``.onnx`` file path.
        ckpt_path: Optional path to a Lightning checkpoint. When given, its
            ``state_dict`` is loaded into ``module`` before export — use this
            to export trained weights rather than ``module``'s current
            (e.g. freshly initialized) ones. Loaded with ``weights_only=True``.
        input_sample: Dummy input tensor for tracing, shape ``(1, C, H, W)``.
            Defaults to ``torch.zeros(1, *module.training_config.example_input_shape)``.
        opset_version: ONNX opset to target. Defaults to ``17`` — a floor
            broadly supported by onnxruntime and TensorRT's ``trtexec``, not
            the newest opset torch happens to default to.

    Raises:
        ImportError: If ``onnx`` is not installed (see the ``export`` extra).
        ValueError: If ``input_sample`` is omitted and
            ``module.training_config.example_input_shape`` is unset.
    """
    _require_onnx()

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        module.load_state_dict(checkpoint["state_dict"])

    if input_sample is None:
        shape = module.training_config.example_input_shape
        if shape is None:
            raise ValueError(
                "to_onnx needs a dummy input to trace the graph: pass "
                "input_sample explicitly, or set "
                "module.training_config.example_input_shape."
            )
        input_sample = torch.zeros(1, *shape)

    model = module.model
    was_training = model.training
    model.eval()
    try:
        with warnings.catch_warnings():
            # dynamo=False (the legacy TorchScript exporter) is deliberate: torch's
            # newer torch.export-based path needs `onnxscript`, outside the
            # `export` extra's onnx+onnxruntime scope (see pyproject.toml). Both
            # warnings below are expected consequences of that choice, not export
            # defects — the DeprecationWarning is the legacy exporter announcing
            # itself, and the TracerWarning is `clamp_output`'s constant-False
            # default branch (SRModel.forward's direct-call convenience, never
            # taken from this call site) being recorded as a Python bool, not a
            # tensor, so it can never diverge across export vs. real inference.
            # Not filtered by `module=`: torch.onnx.export's own DeprecationWarning
            # uses stacklevel=2, so warnings attributes it to *this* call site, not
            # torch.onnx — scoping to this narrow `with` block is what keeps the
            # ignore from hiding anything else.
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(
                model,
                (input_sample,),
                str(file_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {2: "height", 3: "width"},
                    "output": {2: "height", 3: "width"},
                },
                opset_version=opset_version,
                dynamo=False,
            )
    finally:
        model.train(was_training)
