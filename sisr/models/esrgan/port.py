"""Load the authors' released ESRGAN weights (``RRDB_PSNR_x4.pth`` / ``RRDB_ESRGAN_x4.pth``).

A plain PyTorch ``state_dict`` of the ``RRDBNet`` in the authors' repository. Only the names
differ from :class:`~sisr.models.esrgan.RRDBNet`; the tensors are used as they are. **Load
after ``SRLightning`` is constructed**: a config with ``init_strategy='paper'`` re-initialises
the model in its constructor and would overwrite what was loaded before it.
"""

import re
from pathlib import Path

import torch

_FIXED = {
    "conv_first": "head",
    "trunk_conv": "trunk_conv",
    "HRconv": "hr_conv",
    "conv_last": "tail",
}
_TRUNK = re.compile(r"^RRDB_trunk\.(\d+)\.RDB([123])\.(conv[1-5])\.(weight|bias)$")
_UP = re.compile(r"^upconv(\d+)\.(weight|bias)$")
_FIXED_KEY = re.compile(r"^(conv_first|trunk_conv|HRconv|conv_last)\.(weight|bias)$")


def rename_key(key: str) -> str:
    """Map one official parameter name to this package's.

    ``RRDB_trunk.3.RDB2.conv4.weight`` -> ``trunk.3.rdb2.conv4.weight``; ``upconv1`` ->
    ``up.0``; ``conv_first`` -> ``head``; ``HRconv`` -> ``hr_conv``; ``conv_last`` -> ``tail``.

    Raises:
        ValueError: If the key is not one the released network has.
    """
    if m := _TRUNK.match(key):
        i, rdb, conv, kind = m.groups()
        return f"trunk.{i}.rdb{rdb}.{conv}.{kind}"
    if m := _UP.match(key):
        n, kind = m.groups()
        return f"up.{int(n) - 1}.{kind}"
    if m := _FIXED_KEY.match(key):
        return f"{_FIXED[m.group(1)]}.{m.group(2)}"
    raise ValueError(f"unexpected ESRGAN parameter name {key!r}.")


def load_official(path: str | Path) -> dict[str, torch.Tensor]:
    """Read an official ``.pth`` into a :class:`~sisr.models.esrgan.RRDBNet` state dict.

    Args:
        path: ``RRDB_PSNR_x4.pth`` or ``RRDB_ESRGAN_x4.pth``.

    Returns:
        A state dict for ``RRDBNet(scale=4, num_blocks=23)``.

    Raises:
        ValueError: If the file holds a parameter the network does not have.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {rename_key(k): v for k, v in state.items()}
