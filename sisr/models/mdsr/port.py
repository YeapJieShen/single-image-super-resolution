"""Load the authors' released MDSR weights (``mdsr_baseline-*.pt`` / ``mdsr-*.pt``).

A plain ``state_dict`` of the authors' ``MDSR`` serving x2/x3/x4 from one file. Names differ from
:class:`~sisr.models.mdsr.MDSR`; the mean-shift convs are folded into the model as buffers (checked
and dropped, as for EDSR). **Load after ``SRLightning`` is constructed.**
"""

import re
from pathlib import Path

import torch

from sisr.models.edsr.port import drop_mean_shift

from .model import MDSR

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^pre_process\.(\d+)\.(\d+)\.body\.0\.(weight|bias)$"),
        r"pre_process.\1.\2.conv1.\3",
    ),
    (
        re.compile(r"^pre_process\.(\d+)\.(\d+)\.body\.2\.(weight|bias)$"),
        r"pre_process.\1.\2.conv2.\3",
    ),
    (re.compile(r"^body\.(\d+)\.body\.0\.(weight|bias)$"), r"blocks.\1.conv1.\2"),
    (re.compile(r"^body\.(\d+)\.body\.2\.(weight|bias)$"), r"blocks.\1.conv2.\2"),
    (re.compile(r"^head\.0\.(weight|bias)$"), r"head.\1"),
    (re.compile(r"^tail\.0\.(weight|bias)$"), r"tail.\1"),
)


def rename_key(key: str, num_blocks: int) -> str:
    """Map one official parameter name to this package's.

    ``pre_process.1.0.body.0.weight`` -> ``pre_process.1.0.conv1.weight``;
    ``body.5.body.2.bias`` -> ``blocks.5.conv2.bias``; ``body.<num_blocks>`` -> ``body_conv``;
    ``upsample.2.2.weight`` -> ``up.2.1.weight`` (the upsampler interleaves a parameter-free
    ``PixelShuffle``, so its convs sit at even indices); ``head.0`` -> ``head``; ``tail.0`` ->
    ``tail``.

    Raises:
        ValueError: If the key is not one the released network has.
    """
    for pattern, repl in _PATTERNS:
        if pattern.match(key):
            return pattern.sub(repl, key)
    if m := re.match(r"^body\.(\d+)\.(weight|bias)$", key):
        if int(m.group(1)) == num_blocks:
            return f"body_conv.{m.group(2)}"
    if m := re.match(r"^upsample\.(\d+)\.(\d+)\.(weight|bias)$", key):
        scale_idx, k, kind = m.groups()
        return f"up.{scale_idx}.{int(k) // 2}.{kind}"
    raise ValueError(f"unexpected MDSR parameter name {key!r}.")


def load_official(path: str | Path, model: MDSR) -> dict[str, torch.Tensor]:
    """Read an official ``.pt`` into a state dict for ``model``.

    Args:
        path: ``mdsr_baseline-*.pt`` (16 blocks) or ``mdsr-*.pt`` (80 blocks).
        model: The :class:`~sisr.models.mdsr.MDSR` it is for, built with ``scale=[2, 3, 4]`` --
            the scale order is the checkpoint's index order.

    Returns:
        The renamed parameters, without the mean-shift convs.

    Raises:
        ValueError: If the mean shift is not this model's or a parameter is unknown.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    drop_mean_shift(state, model, path)
    return {rename_key(k, model.hparams["num_blocks"]): v for k, v in state.items()}
