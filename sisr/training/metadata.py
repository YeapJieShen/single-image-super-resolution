"""Shared provenance-metadata builder for checkpoints and ONNX exports.

One function, three sinks: :meth:`~sisr.training.lightning_module.SRLightning.on_save_checkpoint`
(the full resumable ``.ckpt``), :class:`~sisr.training.callbacks.SRWeightsCheckpoint` (the bare
distributable ``.pt``), and :func:`sisr.export.to_onnx` (the ``.onnx``'s ``metadata_props``) all
call :func:`build_metadata` so the three payloads cannot drift apart — mirrors how
``SREvalConfig.psnr_keys`` is the single derivation point for its three consumers.

:func:`build_component_metadata` is a second builder for a bare ``.pt`` that is **not** the SR
model — e.g. a discriminator's weights. Both builders are built on one private :func:`_envelope`
helper (``format``/``kind``/``created``/``versions``/``training``), so the two payload shapes
cannot drift apart either. Each carries a ``kind`` field: ``"sr_model"`` for :func:`build_metadata`,
``"component"`` for :func:`build_component_metadata`. This is an additive field, not a format
change — ``FORMAT_VERSION`` is unchanged, and its absence on a file written before ``kind``
existed means ``"sr_model"``, the only kind that existed then.

:func:`build_metadata` carries ``format``, ``kind``, ``created``, ``versions``, ``model``,
``processor``, ``criterion``, ``io``, ``eval_config``, and ``training`` — it describes the
*generator*. :func:`build_component_metadata` carries ``format``, ``kind``, ``created``,
``versions``, ``component``, ``io``, and ``training`` — it describes one other named component
only. Attaching the former to the latter's weights would describe things the file does not
contain (``io.scale``, ``criterion``, ``eval_config``); this is the silent-wrong-artifact class
this metadata exists to prevent, so the two never share a shape beyond the envelope.

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

#: Bumped whenever the metadata shape changes incompatibly. v2: the distributable
#: artifact became safetensors, so the block is carried as a flat string header
#: rather than a pickled dict, and the `.pt` form stopped existing.
FORMAT_VERSION = "sisr-meta-v2"


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


def _envelope(
    kind: str,
    global_step: int | None,
    epoch: int | None,
    monitor: str | None,
    monitor_value: float | None,
    batch_step: int | None = None,
) -> dict[str, Any]:
    """Fields every sisr artifact carries, whatever it holds.

    The one thing :func:`build_metadata` and :func:`build_component_metadata` build on
    in common, so ``format``/``versions``/``training`` cannot drift apart between them.
    """
    return {
        "format": FORMAT_VERSION,
        "kind": kind,
        "created": datetime.now(UTC).isoformat(),
        "versions": {
            # str(...) matters here: torch.__version__ is a TorchVersion (str
            # subclass), which torch.load(weights_only=True) refuses to unpickle
            # as an unlisted global. Coerce to plain str for every field.
            "sisr": str(sisr.__version__),
            "torch": str(torch.__version__),
            "lightning": str(lightning.__version__),
        },
        # Two step counters exist and both are real. `global_step` is the
        # OPTIMIZER count -- factually what it is, and the name Lightning's own
        # checkpoint payload uses for the same quantity, so redefining it here
        # would put this field in direct contradiction with the one beside it.
        # `batch_step` is the batch-counted axis every `self.log` metric lands
        # on, so it is the one a reader correlates a checkpoint against a curve
        # with. Under automatic optimization they are equal; an adversarial
        # module steps two optimizers per batch and they differ by 2x.
        "training": {
            "global_step": global_step,
            "batch_step": batch_step,
            "epoch": epoch,
            "monitor": monitor,
            "monitor_value": monitor_value,
        },
    }


def build_metadata(
    module: SRLightning,
    *,
    global_step: int | None = None,
    batch_step: int | None = None,
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
        batch_step: Batch-counted step at save time — the axis every logged metric
            uses, and therefore the one to correlate this artifact against a curve.
            Equal to ``global_step`` under automatic optimization; half of it on a
            two-optimizer adversarial run.
        epoch: Training epoch at save time, when known (``None`` otherwise).
        monitor: Name of the metric that triggered this specific save (e.g.
            ``"psnr/val/RGB"``), when the sink is monitor-driven (``None`` otherwise).
        monitor_value: Value of ``monitor`` at save time (``None`` otherwise).

    Returns:
        A dict tree carrying ``format``, ``kind`` (``"sr_model"``), ``created``, ``versions``,
        ``model``, ``processor``, ``criterion``, ``io``, ``eval_config``, and ``training``.
        Contains only ``dict``/``list``/``str``/``int``/``float``/``bool``/``None`` values —
        safe for ``torch.save``/``torch.load(weights_only=True)`` and, per top-level field, for
        JSON-encoding into ONNX ``metadata_props``.

    Raises:
        ValueError: If the scale cannot be resolved from either
            ``training_config.scale`` or the model's own hparams.
    """
    model = module.model
    processor = module.processor

    scale = module.training_config.scale
    if scale is None:
        scale = model.hparams.get("scale")
    if scale is None:
        raise ValueError(
            f"Cannot describe a {type(model).__name__} artifact without a scale: "
            f"training_config.scale is None and the model declares no 'scale' "
            f"hyperparameter. A reader cannot use the file without it -- for a "
            f"pre-upsampled architecture the scale is what they must resize the input "
            f"by before feeding it. Set training_config.scale in your config."
        )

    meta = _envelope("sr_model", global_step, epoch, monitor, monitor_value, batch_step)
    meta["model"] = {
        "class_path": _class_path(model),
        "init_args": _to_plain(model.hparams),
        "variant": model.variant_tag,
    }
    meta["processor"] = {
        "class_path": _class_path(processor),
    }
    meta["criterion"] = {
        "class_path": _class_path(module.criterion),
        "description": module.criterion_description,
    }
    meta["io"] = {
        "scale": scale,
        "input": model.input_contract,
        "input_channels": processor.model_channels,
        "output_range": list(processor.output_range),
        "output_colorspace": processor.output_colorspace,
    }
    meta["eval_config"] = _to_plain(dataclasses.asdict(module.eval_config))
    return meta


def build_component_metadata(
    module: SRLightning,
    attribute: str,
    *,
    global_step: int | None = None,
    batch_step: int | None = None,
    epoch: int | None = None,
    monitor: str | None = None,
    monitor_value: float | None = None,
) -> dict[str, Any]:
    """Provenance for a bare weights file that is **not** the SR model.

    :func:`build_metadata` describes the *generator* — ``io.scale``, ``io.input``,
    ``criterion``, ``eval_config``. Attaching that to a discriminator's weights would
    describe something the file does not contain, which is the silent-wrong-artifact
    failure this metadata exists to prevent.

    ``io.input_range`` is the load-bearing field: a discriminator is trained on
    model-space tensors (``[-1, 1]`` under ``RGBSignedOutputProcessor``), and feeding
    it ``[0, 1]`` data later is wrong with no error.

    Args:
        module: The Lightning module owning the component.
        attribute: Attribute name of the component on ``module`` (e.g. ``'discriminator'``).
        global_step: Optimizer step at save time, when known.
        batch_step: Batch-counted step at save time — see :func:`build_metadata`.
        epoch: Training epoch at save time, when known.
        monitor: Metric that triggered this save, when monitor-driven.
        monitor_value: Value of ``monitor`` at save time.

    Returns:
        A dict tree carrying ``format``, ``kind`` (``"component"``), ``created``,
        ``versions``, ``component``, ``io``, and ``training`` — a plain dict, ``torch.load
        (weights_only=True)``-safe like :func:`build_metadata`.
    """
    component = getattr(module, attribute)
    processor = module.processor
    meta = _envelope("component", global_step, epoch, monitor, monitor_value, batch_step)
    meta["component"] = {
        "name": attribute,
        "class_path": _class_path(component),
        "init_args": _to_plain(getattr(component, "hparams", {})),
        # Duck-typed: a co-trained component need not be an SRModel, and one
        # without a tag simply contributes nothing to the filename.
        "variant": getattr(component, "variant_tag", None),
    }
    meta["io"] = {
        "input_range": list(processor.output_range),
        "input_colorspace": processor.output_colorspace,
    }
    return meta
