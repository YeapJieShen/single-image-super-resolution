"""Test-time self-ensemble ("x8") — average a model's output over its 8 dihedral transforms.

Geometric self-ensembling (used by EDSR/RCAN/many SR papers to report a "+"-suffixed score)
feeds a model each of the 8 symmetries of the square — the 4 rotations and their mirror
images — and averages the outputs after undoing each transform, so every output lines back up
in the original orientation before the mean is taken.

Built from exactly 3 *involutions* — horizontal flip, vertical flip, and transpose (swap H/W)
— applied in a fixed order to make a transform, and the SAME 3 ops in the REVERSE order to
undo it. Composing self-inverse functions in reverse order is what inverts a composition
(``(f3∘f2∘f1)⁻¹ = f1⁻¹∘f2⁻¹∘f3⁻¹``, and each ``fi⁻¹ == fi`` here), so the round trip holds by
construction for all 8 combinations — never re-derive rotation angles by hand.

Depends on ``torch`` only and is a pure function of ``model_fn`` and the input tensor — no
``SRLightning``, no ``eval_config`` read here. Callers decide when this runs (see
``SRLightning._forward_lr``'s own ``self_ensemble`` argument); this module only implements
what running it means.
"""

import itertools
from collections.abc import Callable
from typing import cast

import torch

# The 8 elements of the square's symmetry group (dihedral order 8), as
# (hflip, vflip, transpose) flag combinations. Which 8 combinations doesn't matter for
# correctness -- fixed here (rather than re-generated per call) only so the count is
# obviously 8 and tests can iterate a known set.
_DIHEDRAL_FLAGS: tuple[tuple[bool, bool, bool], ...] = cast(
    tuple[tuple[bool, bool, bool], ...], tuple(itertools.product([False, True], repeat=3))
)


def _transform(x: torch.Tensor, hflip: bool, vflip: bool, transpose: bool) -> torch.Tensor:
    """Apply hflip, then vflip, then transpose (each conditional) to x's last 2 dims."""
    if hflip:
        x = torch.flip(x, dims=[-1])
    if vflip:
        x = torch.flip(x, dims=[-2])
    if transpose:
        x = x.transpose(-2, -1).contiguous()
    return x


def _untransform(x: torch.Tensor, hflip: bool, vflip: bool, transpose: bool) -> torch.Tensor:
    """Undo `_transform`'s exact (hflip, vflip, transpose) call — reverse order, same ops."""
    if transpose:
        x = x.transpose(-2, -1).contiguous()
    if vflip:
        x = torch.flip(x, dims=[-2])
    if hflip:
        x = torch.flip(x, dims=[-1])
    return x


def self_ensemble_forward(
    model_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """Average ``model_fn(x)`` over the 8 dihedral (flip/rotation) transforms of ``x``.

    For each of the 8 transforms: transform ``x``, run ``model_fn``, inverse-transform the
    result back to ``x``'s original orientation. The 8 inverse-transformed outputs are then
    averaged. ``model_fn`` is called exactly 8 times — 8x the compute of a single pass.

    Args:
        model_fn: Called once per transform with a tensor shaped like a transform of ``x``
            — typically ``self.model`` or ``self._compiled``. Must accept any H/W (a plain
            conv stack does); a model hardcoding ``x``'s spatial shape cannot be
            self-ensembled, since 4 of the 8 transforms swap H and W.
        x: Input tensor, spatial dims last (``..., H, W``); otherwise unconstrained (batch/
            channel dims pass through untouched, and H need not equal W).

    Returns:
        Tensor the same shape ``model_fn(x)`` alone would return, averaged over the 8 passes.
    """
    outputs = [_untransform(model_fn(_transform(x, *flags)), *flags) for flags in _DIHEDRAL_FLAGS]
    return torch.stack(outputs, dim=0).mean(dim=0)
