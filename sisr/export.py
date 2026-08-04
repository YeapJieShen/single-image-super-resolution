"""Bare-model ONNX export.

Exports ``module.model`` — the wrapped :class:`~sisr.models.base.SRModel`,
*without* the :class:`~sisr.processors.SRProcessor` — so the graph matches
:meth:`SRLightning.forward` exactly (which is itself model-only; see
``sisr/training/lightning_module.py``). This is a deliberate choice, not an
oversight: a consumer of the exported graph is expected to have ``sisr``
importable and call ``processor.extract`` / ``processor.reconstruct``
themselves.

The exported graph also carries the same provenance metadata as the checkpoint
sinks (:func:`sisr.training.metadata.build_metadata`), written one field per
``onnx.ModelProto.metadata_props`` entry rather than a single opaque JSON blob —
Netron renders per-field props as a readable table, which is most of the value.
``metadata_props`` is string->string only, so non-string fields are JSON-encoded.

For :class:`~sisr.processors.RGBProcessor` architectures (SRResNet), those
methods are identity functions, so the bare graph is already an end-to-end
LR-RGB -> SR-RGB pipeline. Under
:class:`~sisr.processors.RGBSignedOutputProcessor` the same graph emits
``[-1, 1]`` and the consumer must apply ``(out + 1) / 2``; the input side is
unchanged. For :class:`~sisr.processors.YChannelProcessor`
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

import json
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
    height/width) because real images vary in H/W independently of
    ``training_config.example_input_shape``: that field is a representative
    dummy shared by several fixed-shape uses (TensorBoard graph, ModelSummary
    FLOPs, the compile warm-up, and this function's own default
    ``input_sample``), none of which constrain inference spatial dims.

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

    global_step: int | None = None
    epoch: int | None = None
    monitor: str | None = None
    monitor_value: float | None = None
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        module.load_state_dict(checkpoint["state_dict"])
        # Carry the source checkpoint's own provenance forward when present (only
        # populated on checkpoints saved after SRLightning.on_save_checkpoint shipped;
        # absent on older ones, in which case these stay None).
        global_step = checkpoint.get("global_step")
        epoch = checkpoint.get("epoch")
        ckpt_training = checkpoint.get("sisr_meta", {}).get("training", {})
        monitor = ckpt_training.get("monitor")
        monitor_value = ckpt_training.get("monitor_value")

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

    _write_metadata_props(
        module,
        file_path,
        global_step=global_step,
        epoch=epoch,
        monitor=monitor,
        monitor_value=monitor_value,
    )


def _write_metadata_props(
    module: SRLightning,
    file_path: str | PathLike,
    *,
    global_step: int | None,
    epoch: int | None,
    monitor: str | None,
    monitor_value: float | None,
) -> None:
    """Stamp the shared provenance metadata onto the just-exported ONNX file.

    Loads the model ``torch.onnx.export`` just wrote, adds one
    ``metadata_props`` entry per top-level :func:`~sisr.training.metadata.build_metadata`
    field, and re-saves in place. ``metadata_props`` is ``string -> string`` only, so
    ``str`` fields (``format``, ``created``) are stored as-is and every other
    (dict-valued) field is JSON-encoded — per-field rather than one opaque blob, so a
    tool like Netron renders them as a readable table.

    Args:
        module: The :class:`~sisr.training.SRLightning` instance that was exported.
        file_path: Path to the ``.onnx`` file just written by ``torch.onnx.export``.
        global_step: Forwarded to :func:`~sisr.training.metadata.build_metadata`.
        epoch: Forwarded to :func:`~sisr.training.metadata.build_metadata`.
        monitor: Forwarded to :func:`~sisr.training.metadata.build_metadata`.
        monitor_value: Forwarded to :func:`~sisr.training.metadata.build_metadata`.
    """
    import onnx

    from .training.metadata import build_metadata

    meta = build_metadata(
        module,
        global_step=global_step,
        epoch=epoch,
        monitor=monitor,
        monitor_value=monitor_value,
    )
    onnx_model = onnx.load(str(file_path))
    for key, value in meta.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value if isinstance(value, str) else json.dumps(value)
    onnx.save(onnx_model, str(file_path))
