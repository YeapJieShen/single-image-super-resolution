"""Shared provenance-metadata builder for checkpoints and ONNX exports.

One function, three sinks: :meth:`~sisr.training.lightning_module.SRLightning.on_save_checkpoint`
(the full resumable ``.ckpt``), :class:`~sisr.training.callbacks.SRWeightsCheckpoint` (the bare
distributable ``.pt``), and :func:`sisr.export.to_onnx` (the ``.onnx``'s ``metadata_props``) all
call :func:`build_metadata` so the three payloads cannot drift apart — mirrors how
``SREvalConfig.psnr_keys`` is the single derivation point for its three consumers.

Carries ``format``, ``created``, ``versions``, ``model``, ``processor``, ``criterion``, ``io``,
``eval_config``, and ``training``.

Deliberately omits dataset paths: Ultralytics' ``train_args`` leaks local filesystem layout into
distributed files; this stays leak-free by construction. If dataset provenance is ever added,
it must be names only, never paths.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import lightning
import torch

import sisr

if TYPE_CHECKING:
    from .lightning_module import SRLightning

#: Bumped whenever the metadata shape changes incompatibly.
FORMAT_VERSION = "sisr-meta-v1"


def _to_plain(obj: Any) -> Any:
    """Recursively convert tuples to lists for JSON- and weights_only-safety."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    return obj


def _class_path(obj: Any) -> str:
    """Dotted ``module.ClassName`` path for an instance's class — the YAML's own shape."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def build_metadata(
    module: SRLightning,
    *,
    global_step: int | None = None,
    epoch: int | None = None,
    monitor: str | None = None,
    monitor_value: float | None = None,
) -> dict[str, Any]:
    """Build the provenance-metadata dict shared by every distributable sisr artifact.

    Args:
        module: The :class:`~sisr.training.lightning_module.SRLightning` instance to
            describe — supplies ``model``, ``processor``, ``training_config``, and
            ``eval_config``.
        global_step: Optimizer step at save time, when known (``None`` otherwise).
        epoch: Training epoch at save time, when known (``None`` otherwise).
        monitor: Name of the metric that triggered this specific save (e.g.
            ``"psnr/val/RGB"``), when the sink is monitor-driven (``None`` otherwise).
        monitor_value: Value of ``monitor`` at save time (``None`` otherwise).

    Returns:
        A dict tree carrying ``format``, ``created``, ``versions``, ``model``, ``processor``,
        ``criterion``, ``io``, ``eval_config``, and ``training``. Contains only ``dict``/
        ``list``/``str``/``int``/``float``/``bool``/``None`` values — safe for ``torch.save``/
        ``torch.load(weights_only=True)`` and, per top-level field, for JSON-encoding into
        ONNX ``metadata_props``.
    """
    model = module.model
    processor = module.processor

    scale = module.training_config.scale
    if scale is None:
        scale = model.hparams.get("scale")

    return {
        "format": FORMAT_VERSION,
        "created": datetime.now(UTC).isoformat(),
        "versions": {
            # str(...) matters here: torch.__version__ is a TorchVersion (str
            # subclass), which torch.load(weights_only=True) refuses to unpickle
            # as an unlisted global. Coerce to plain str for every field.
            "sisr": str(sisr.__version__),
            "torch": str(torch.__version__),
            "lightning": str(lightning.__version__),
        },
        "model": {
            "class_path": _class_path(model),
            "init_args": _to_plain(model.hparams),
        },
        "processor": {
            "class_path": _class_path(processor),
        },
        "criterion": {
            "class_path": _class_path(module.criterion),
            "description": module.criterion_description,
        },
        "io": {
            "scale": scale,
            "input": model.input_contract,
            "input_channels": processor.model_channels,
            "output_range": list(processor.output_range),
            "output_colorspace": processor.output_colorspace,
        },
        "eval_config": _to_plain(dataclasses.asdict(module.eval_config)),
        "training": {
            "global_step": global_step,
            "epoch": epoch,
            "monitor": monitor,
            "monitor_value": monitor_value,
        },
    }
