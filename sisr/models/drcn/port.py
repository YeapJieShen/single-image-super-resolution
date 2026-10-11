"""Load the authors' released DRCN weights (MatConvNet ``DRCN_sf{2,3,4}.mat``).

The release is a MatConvNet DagNN. Its wiring, which :class:`~sisr.models.drcn.DRCN`
reproduces: ``conv1``/``conv2`` (embedding), ``conv_outh*`` (one shared 3x3 conv,
here ``recon1``, applied once before the first prediction and again inside every
recursion), ``conv3..conv18`` (one shared inference conv, here ``infer``),
``conv_out*`` (one shared 3x3 conv to a single channel, here ``recon2``), and a
``1x1x17`` ensemble. The DropOut layers are training-only and are ignored.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

#: MatConvNet parameter name -> (DRCN state_dict key, has the singleton channel squeezed).
_PARAMS: dict[str, str] = {
    "filters1": "embed.0.weight",
    "biases1": "embed.0.bias",
    "filters2": "embed.2.weight",
    "biases2": "embed.2.bias",
    "filters19": "recon1.weight",
    "biases19": "recon1.bias",
    "filters_share": "infer.weight",
    "biases_share": "infer.bias",
    "filters20": "recon2.weight",
    "biases20": "recon2.bias",
    "ensemble_weight": "ensemble",
}


def load_matconvnet(path: str | Path) -> dict[str, torch.Tensor]:
    """Read an official ``DRCN_sf*.mat`` into a :class:`~sisr.models.drcn.DRCN` state dict.

    MatConvNet stores a conv filter as ``(H, W, C_in, C_out)`` and squeezes
    singleton trailing dimensions on load; PyTorch wants ``(C_out, C_in, H, W)``.
    Both libraries compute cross-correlation, so no kernel flip is needed.

    Args:
        path: The ``.mat`` file.

    Returns:
        A state dict for ``DRCN(in_out_channels=1, features=256, recursions=16)``.

    Raises:
        ImportError: If SciPy is not installed (``pip install 'sisr[weights]'``).
        ValueError: If the file lacks an expected parameter.
    """
    try:
        from scipy.io import loadmat
    except ImportError as e:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "Reading MatConvNet weights needs SciPy: pip install 'sisr[weights]'"
        ) from e

    net = loadmat(str(path), squeeze_me=True, struct_as_record=False)["net"]
    params: dict[str, Any] = {p.name: np.asarray(p.value) for p in net.params}
    missing = sorted(set(_PARAMS) - set(params))
    if missing:
        raise ValueError(f"{path} lacks the expected DRCN parameters: {missing}")

    state: dict[str, torch.Tensor] = {}
    for mat_name, key in _PARAMS.items():
        value = params[mat_name].astype(np.float32)
        if key.endswith(".weight"):
            if value.ndim == 3:  # a singleton C_in (conv1) or C_out (conv_out) was squeezed
                value = value[..., None, :] if key == "embed.0.weight" else value[..., None]
            value = value.transpose(3, 2, 0, 1)
        elif key.endswith(".bias"):
            value = value.reshape(-1)
        state[key] = torch.from_numpy(np.ascontiguousarray(value))
    return state
