"""Load the authors' released EDSR weights (``edsr_baseline_x*.pt`` / ``edsr_x*.pt``).

A plain ``state_dict`` of the authors' ``EDSR``. The names differ from
:class:`~sisr.models.edsr.EDSR` and the two mean-shift convs (``sub_mean``/``add_mean``, fixed
1x1 identity convs with the dataset mean as bias, on the 0-255 range) are folded into the model
as buffers: the port checks they equal what the model computes and drops them. **Load after
``SRLightning`` is constructed** -- a config with ``init_strategy='paper'`` re-initialises the
model in its constructor.
"""

import re
from pathlib import Path

import torch

from .model import EDSR

_BLOCK = re.compile(r"^body\.(\d+)\.body\.([02])\.(weight|bias)$")
_UP = re.compile(r"^tail\.0\.(\d+)\.(weight|bias)$")


def rename_key(key: str, num_blocks: int) -> str:
    """Map one official parameter name to this package's.

    ``body.3.body.2.weight`` -> ``blocks.3.conv2.weight``; ``body.<num_blocks>`` -> ``body_conv``;
    ``head.0`` -> ``head``; ``tail.0.2`` -> ``up.1`` (the upsampler interleaves a parameter-free
    ``PixelShuffle``, so its convs sit at even indices); ``tail.1`` -> ``tail``.

    Raises:
        ValueError: If the key is not one the released network has.
    """
    if m := _BLOCK.match(key):
        i, which, kind = m.groups()
        return f"blocks.{i}.conv{1 if which == '0' else 2}.{kind}"
    if m := re.match(r"^body\.(\d+)\.(weight|bias)$", key):
        i, kind = m.groups()
        if int(i) != num_blocks:
            raise ValueError(f"unexpected EDSR parameter name {key!r}.")
        return f"body_conv.{kind}"
    if m := re.match(r"^head\.0\.(weight|bias)$", key):
        return f"head.{m.group(1)}"
    if m := _UP.match(key):
        idx, kind = m.groups()
        return f"up.{int(idx) // 2}.{kind}"
    if m := re.match(r"^tail\.1\.(weight|bias)$", key):
        return f"tail.{m.group(1)}"
    raise ValueError(f"unexpected EDSR parameter name {key!r}.")


def load_official(path: str | Path, model: EDSR) -> dict[str, torch.Tensor]:
    """Read an official ``.pt`` into a state dict for ``model``.

    Args:
        path: One of the six released checkpoints.
        model: The :class:`~sisr.models.edsr.EDSR` it is for -- its block count tells the
            trunk's closing conv from a residual block, and its mean shift is what the
            checkpoint's must equal.

    Returns:
        The renamed parameters, without the mean-shift convs.

    Raises:
        ValueError: If the checkpoint's mean shift is not this model's (a different dataset
            mean or range), or it holds a parameter the network does not have.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    rng = model.rgb_range
    mean = model.mean.flatten() * rng
    eye = torch.eye(3).view(3, 3, 1, 1)
    for name, sign in (("sub_mean", -1.0), ("add_mean", 1.0)):
        if not (
            torch.allclose(state.pop(f"{name}.weight"), eye, atol=1e-6)
            and torch.allclose(state.pop(f"{name}.bias"), sign * mean, atol=1e-3)
        ):
            raise ValueError(
                f"{Path(path).name}: its {name} is not the identity shifted by "
                f"{sign:+g} * rgb_mean * rgb_range for this model (rgb_mean="
                f"{model.hparams['rgb_mean']}, rgb_range={rng})."
            )
    return {rename_key(k, model.hparams["num_blocks"]): v for k, v in state.items()}
