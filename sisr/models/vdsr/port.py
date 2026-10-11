"""Load the authors' released VDSR weights (MatConvNet ``VDSR.mat``, a ``simplenn``).

The release is a MatConvNet ``simplenn``: 20 same-padded 3x3 convs with ReLUs between
them and a euclidean loss, which :class:`~sisr.models.vdsr.VDSR` reproduces (the loss
is training-only and ignored). **Its parameters are not readable by ``scipy.io.loadmat``
as such**: the authors saved the net from the GPU, so every filter, bias and momentum
is a MATLAB ``gpuArray`` object, and ``loadmat`` returns only an opaque reference to
each. The numbers live in the file's ``__function_workspace__`` -- an embedded MAT-file
stream holding the objects' data as a cell array. This module walks that stream itself.

The cell holds, per layer in order, four float32 arrays: filters, biases, and their two
momentum buffers (``object id k`` is slot ``k + 1``, and ids run
``4 * layer + (1, 2, 3, 4)``). Only the first two are weights.
"""

import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

_MI_MATRIX = 14
_MI_SINGLE = 7
_MX_CELL, _MX_STRUCT, _MX_OPAQUE = 1, 2, 17

#: ``(MATLAB dims, flat column-major data)`` of one float32 array.
_Leaf = tuple[tuple[int, ...], np.ndarray]


def _tag(b: bytes, o: int) -> tuple[int, int, int, int]:
    """``(type, nbytes, data offset, next element offset)`` of the MAT5 element at ``o``."""
    t, n = struct.unpack_from("<II", b, o)
    if t >> 16:  # small data element: size packed into the type word, data inline
        return t & 0xFFFF, t >> 16, o + 4, o + 8
    return t, n, o + 8, o + 8 + ((n + 7) // 8) * 8


def _single_arrays(b: bytes, o: int, end: int, out: list[_Leaf]) -> None:
    """Appends every float32 array reachable from the elements in ``b[o:end]``, in file order."""
    while o < end:
        t, n, d, nxt = _tag(b, o)
        if t == _MI_MATRIX:
            _matrix(b, d, d + n, out)
        o = nxt


def _matrix(b: bytes, o: int, end: int, out: list[_Leaf]) -> None:
    _, _, d, o = _tag(b, o)
    cls = struct.unpack_from("<I", b, d)[0] & 0xFF
    _, n, d, o = _tag(b, o)
    dims = struct.unpack_from(f"<{n // 4}i", b, d)
    _, _, _, o = _tag(b, o)  # the array's name
    if cls == _MX_CELL:
        for _ in range(int(np.prod(dims))):
            t, n, d, nxt = _tag(b, o)
            _matrix(b, d, d + n, out)
            o = nxt
    elif cls == _MX_STRUCT:
        _, _, d, o = _tag(b, o)
        field_len = struct.unpack_from("<i", b, d)[0]
        _, n, _, o = _tag(b, o)
        for _ in range(int(np.prod(dims)) * (n // field_len)):
            t, n, d, nxt = _tag(b, o)
            _matrix(b, d, d + n, out)
            o = nxt
    elif cls == _MX_OPAQUE:
        _single_arrays(b, o, end, out)
    else:
        t, n, d, _ = _tag(b, o)
        if t == _MI_SINGLE:
            out.append((tuple(dims), np.frombuffer(b, dtype="<f4", count=n // 4, offset=d)))


def mcos_single_arrays(workspace: bytes) -> list[_Leaf]:
    """Every float32 array in a MAT-file function workspace, as ``(MATLAB dims, flat data)``.

    Args:
        workspace: The ``__function_workspace__`` bytes, after their 8-byte header.

    Returns:
        The arrays in file order. ``flat`` is MATLAB's column-major storage: reshape with
        ``order='F'``.
    """
    out: list[_Leaf] = []
    _single_arrays(workspace, 0, len(workspace), out)
    return out


def state_dict_from_arrays(
    arrays: list[_Leaf],
) -> dict[str, torch.Tensor]:
    """Map the workspace's per-layer ``(filters, biases, momenta...)`` arrays to a state dict.

    MatConvNet stores a filter as ``(H, W, C_in, C_out)`` (dropping a trailing singleton
    dimension on save); PyTorch wants ``(C_out, C_in, H, W)``. Both libraries compute
    cross-correlation, so no kernel flip is needed.

    Args:
        arrays: What :func:`mcos_single_arrays` returned.

    Returns:
        A state dict for ``VDSR(depth=len(arrays) // 4)``.

    Raises:
        ValueError: If the arrays are not four per layer, or a layer's shapes disagree.
    """
    if not arrays or len(arrays) % 4:
        raise ValueError(
            f"expected 4 float32 arrays per layer (filters, biases, 2 momenta); got {len(arrays)}."
        )
    state: dict[str, torch.Tensor] = {}
    for layer in range(len(arrays) // 4):
        (fdims, fdata), (bdims, bdata) = arrays[4 * layer], arrays[4 * layer + 1]
        h, w, c_in, c_out = (tuple(fdims) + (1, 1))[:4]
        if fdata.size != h * w * c_in * c_out or bdata.size != c_out:
            raise ValueError(
                f"layer {layer}: filters {fdims} / biases {bdims} do not fit one conv."
            )
        weight = fdata.reshape((h, w, c_in, c_out), order="F").transpose(3, 2, 0, 1)
        state[f"body.{2 * layer}.weight"] = torch.from_numpy(np.ascontiguousarray(weight))
        state[f"body.{2 * layer}.bias"] = torch.from_numpy(bdata.copy())
    return state


def load_matconvnet(path: str | Path) -> dict[str, torch.Tensor]:
    """Read the official ``VDSR.mat`` into a :class:`~sisr.models.vdsr.VDSR` state dict.

    Args:
        path: The ``.mat`` file.

    Returns:
        A state dict for ``VDSR(in_out_channels=1, depth=20, features=64)``.

    Raises:
        ImportError: If SciPy is not installed (``pip install 'sisr[weights]'``).
        ValueError: If the file has no function workspace or its arrays do not fit.
    """
    try:
        from scipy.io import loadmat
    except ImportError as e:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "Reading MatConvNet weights needs SciPy: pip install 'sisr[weights]'"
        ) from e

    contents: dict[str, Any] = loadmat(str(path))
    workspace = contents.get("__function_workspace__")
    if workspace is None:
        raise ValueError(f"{path} has no __function_workspace__: not a GPU-saved MatConvNet net.")
    return state_dict_from_arrays(mcos_single_arrays(workspace.tobytes()[8:]))
