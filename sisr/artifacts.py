"""The distributable weights artifact: one owner for writing, reading and checking it.

A run produces two kinds of file. The resumable ``.ckpt`` is Lightning's, and
stays as it is. This module owns the other: the bare, optimizer-free weights
file handed to somebody else. Writing, reading and validation live here so the
payload's shape has one owner rather than being re-derived per reader.

**Why safetensors.** Not because our readers are unsafe -- every ``torch.load``
here passes ``weights_only=True`` against a pinned torch floor. The exposure is
a *consumer* loading a published file without it, or on an older torch.
safetensors removes that by construction rather than by mitigation, and it is
what the surrounding ecosystem reads.

**The header is flat.** safetensors metadata is ``str -> str``, so the block is
written one entry per top-level field, non-strings JSON-encoded. Not a single
blob, deliberately: per-field survives inspection by anything that can open the
file, which is most of the value. The ONNX sink shares this encoder for the
same reason.
"""

from __future__ import annotations

import json
import warnings
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

#: Extension for every artifact this module writes.
SUFFIX = ".safetensors"

#: Fields whose disagreement changes what the model's output *means*, rather than
#: merely describing the run that produced it. Getting one of these wrong yields a
#: plausible image and no error, which is the failure this validation exists for.
MEANING_FIELDS: tuple[tuple[str, str], ...] = (
    ("processor", "class_path"),
    ("io", "output_range"),
)


def encode_metadata(meta: dict[str, Any]) -> dict[str, str]:
    """Flatten a provenance block into a ``str -> str`` header.

    Args:
        meta: A block from :func:`~sisr.training.metadata.build_metadata` or
            :func:`~sisr.training.metadata.build_component_metadata`.

    Returns:
        One entry per top-level field. String values are stored as-is so they
        stay readable; everything else is JSON-encoded.
    """
    return {k: v if isinstance(v, str) else json.dumps(v) for k, v in meta.items()}


def decode_metadata(header: dict[str, str] | None) -> dict[str, Any]:
    """Rebuild a provenance block from a flat header.

    Args:
        header: The header as stored, or ``None`` for a file carrying none.

    Returns:
        The nested block. A value that is not valid JSON is returned as the
        string it is, which is what keeps the plain-string fields round-tripping.
    """
    out: dict[str, Any] = {}
    for key, value in (header or {}).items():
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def save(path: str | PathLike, tensors: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    """Write one component's weights plus its provenance.

    Args:
        path: Destination. The caller owns the filename; this owns the directory.
        tensors: A component's ``state_dict()``.
        meta: Provenance to carry in the header.
    """
    # Lightning creates a checkpoint's directory inside its own IO plugin, and
    # SRWeightsCheckpoint._save_checkpoint deliberately bypasses that path to write a
    # different payload -- so nothing was creating this one. It worked only while a sibling
    # SRCheckpoint happened to run first and make the directory; a weights-only
    # configuration failed on its first save.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # safetensors stores raw buffers and so refuses a view; a state_dict is free
    # to hand back non-contiguous tensors.
    save_file(
        {k: v.detach().contiguous() for k, v in tensors.items()},
        str(path),
        metadata=encode_metadata(meta),
    )


def load(path: str | PathLike) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read an artifact written by :func:`save`.

    Args:
        path: The file to read.

    Returns:
        ``(tensors, meta)``.

    Raises:
        ValueError: If the file carries no provenance. Loading weights whose
            provenance is unknown is what every check downstream exists to
            prevent, so an unlabelled file is refused rather than guessed at.
    """
    from safetensors import safe_open

    path = Path(path)
    with safe_open(str(path), framework="pt") as handle:
        meta = decode_metadata(handle.metadata())
    if not meta:
        raise ValueError(
            f"{path.name!r} carries no sisr provenance header, so nothing can be checked "
            f"about it -- not the architecture, the processor, the output range, or the "
            f"upscaling factor. Weights that load into a mismatched model train and score "
            f"without erroring; only the numbers are wrong. Point at a file this project "
            f"wrote."
        )
    return load_file(str(path)), meta


def require_compatible(
    found: dict[str, Any],
    expected: dict[str, Any],
    *,
    fields: tuple[tuple[str, str], ...] = MEANING_FIELDS,
    source: str = "artifact",
) -> None:
    """Refuse a mismatch that changes meaning; warn about one that only describes.

    Args:
        found: Provenance read from the file.
        expected: Provenance describing the run that wants to use it.
        fields: ``(section, key)`` pairs to refuse on. Defaults to
            :data:`MEANING_FIELDS`; a caller with more to lose passes more.
        source: Name used in messages, normally the filename.

    Raises:
        ValueError: If any of ``fields`` disagrees.
    """
    for section, key in fields:
        want = expected.get(section, {}).get(key)
        got = found.get(section, {}).get(key)
        if got == want:
            continue
        raise ValueError(
            f"{source} was written by a run whose {section}.{key} is {got!r}, but this "
            f"run's is {want!r}. Weights used under a different architecture, processor, "
            f"output range or upscaling factor train and score without ever erroring -- "
            f"only the numbers are wrong. Point at a matching artifact, or align this "
            f"run's config with the one that produced it."
        )

    # Version drift cannot change what the tensors mean, so it is said once and
    # not enforced -- refusing here would make every artifact expire on upgrade.
    found_versions = found.get("versions", {})
    expected_versions = expected.get("versions", {})
    drifted = {
        name: (found_versions.get(name), value)
        for name, value in expected_versions.items()
        if found_versions.get(name) != value
    }
    if drifted:
        detail = ", ".join(f"{n}: {was} -> {now}" for n, (was, now) in sorted(drifted.items()))
        warnings.warn(
            f"{source} was written under different library versions ({detail}). The "
            f"tensors are loaded as-is; this is provenance drift, not a mismatch.",
            UserWarning,
            stacklevel=2,
        )


def stem(meta: dict[str, Any]) -> str:
    """Filename identity for an artifact, derived from its own provenance.

    Deliberately a projection of the metadata rather than a second description
    built from the module: a filename and a header that disagree is precisely the
    class of defect this project keeps finding, and deriving one from the other
    makes it unrepresentable.

    ``SRResNet_x4_RGB_16B64F`` — architecture, scale, colourspace, variant. A
    component drops scale and colourspace, neither of which describes a critic:
    ``SRDiscriminator_96``.

    Args:
        meta: A block from :func:`~sisr.training.metadata.build_metadata` or
            :func:`~sisr.training.metadata.build_component_metadata`.

    Returns:
        The identity, with any absent part simply omitted rather than rendered
        as ``None``.
    """
    if meta.get("kind") == "component":
        block = meta.get("component", {})
        parts = [block.get("class_path", "").rsplit(".", 1)[-1], block.get("variant")]
    else:
        block = meta.get("model", {})
        io = meta.get("io", {})
        scale = io.get("scale")
        parts = [
            block.get("class_path", "").rsplit(".", 1)[-1],
            f"x{scale}" if scale is not None else None,
            io.get("output_colorspace"),
            block.get("variant"),
        ]
    return "_".join(p for p in parts if p)
