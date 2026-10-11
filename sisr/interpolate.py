"""Network interpolation (ESRGAN, Wang et al. 2018, Section 3.4).

Blends two weight artifacts of one architecture, ``(1 - alpha) * A + alpha * B``, to trade
fidelity (the PSNR-trained network) against texture (the GAN-trained one) without retraining.
A plain function, not a console script: it is a few lines over :mod:`sisr.artifacts` and a
command-line surface is a design decision this project has not made.

>>> from sisr.interpolate import interpolate_weights
>>> interpolate_weights("psnr.safetensors", "gan.safetensors", 0.8, "blend.safetensors")
"""

from os import PathLike
from typing import Any

import torch

from . import artifacts

#: Header sections that identify an architecture and what its output means. Two artifacts
#: that disagree on any of them are not blendable.
_IDENTITY: tuple[tuple[str, str], ...] = (
    ("model", "class_path"),
    ("model", "init_args"),
    ("processor", "class_path"),
    ("io", "scale"),
    ("io", "output_range"),
)


def interpolate_weights(
    path_a: str | PathLike,
    path_b: str | PathLike,
    alpha: float,
    out: str | PathLike,
) -> None:
    """Write the linear blend ``(1 - alpha) * A + alpha * B`` of two weight artifacts.

    Args:
        path_a: The ``alpha = 0`` end (typically the PSNR-oriented weights).
        path_b: The ``alpha = 1`` end (typically the GAN-trained weights).
        alpha: Blend weight of ``path_b``. ``[0, 1]`` interpolates; outside it extrapolates, which
            is allowed but rarely wanted.
        out: Destination artifact. Carries A's header plus an ``interpolation`` record
            ``{a, b, alpha}`` naming the inputs by filename.

    Raises:
        ValueError: If either file is not a generator artifact; if the two disagree on
            architecture, hyperparameters, processor, scale or output range (naming the
            differing keys); or if their tensors do not match in names, shapes and dtypes.
    """
    tensors_a, meta_a = artifacts.load(path_a)
    tensors_b, meta_b = artifacts.load(path_b)
    for name, found in (("a", meta_a), ("b", meta_b)):
        if found.get("kind") == "component":
            raise ValueError(f"{name} is a component artifact, not a generator's weights.")

    differing = []
    for section, key in _IDENTITY:
        in_a, in_b = meta_a.get(section, {}).get(key), meta_b.get(section, {}).get(key)
        if in_a != in_b:
            differing.append(f"{section}.{key}: {in_a!r} vs {in_b!r}")
    if differing:
        raise ValueError(
            "cannot interpolate artifacts of different networks; they differ in "
            + "; ".join(differing)
        )
    if tensors_a.keys() != tensors_b.keys():
        raise ValueError(
            f"tensor names differ: only in a {sorted(tensors_a.keys() - tensors_b.keys())}, "
            f"only in b {sorted(tensors_b.keys() - tensors_a.keys())}."
        )
    blended: dict[str, torch.Tensor] = {}
    for name, a in tensors_a.items():
        b = tensors_b[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(
                f"{name}: a is {a.dtype} {tuple(a.shape)}, b is {b.dtype} {tuple(b.shape)}."
            )
        blended[name] = ((1 - alpha) * a.double() + alpha * b.double()).to(a.dtype)

    header: dict[str, Any] = dict(meta_a)
    header["interpolation"] = {
        "a": str(getattr(path_a, "name", path_a)).rsplit("/", 1)[-1],
        "b": str(getattr(path_b, "name", path_b)).rsplit("/", 1)[-1],
        "alpha": alpha,
    }
    artifacts.save(out, blended, header)
