"""Pixel-domain losses: Charbonnier distance and total-variation regularisation."""

import torch

_REDUCTIONS = ("mean", "sum", "none")


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply a torch-style reduction to an elementwise loss tensor."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def _check_reduction(reduction: str) -> str:
    """Validate a reduction name at construction, not at the first step."""
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}; got {reduction!r}")
    return reduction


class CharbonnierLoss(torch.nn.Module):
    """Charbonnier distance ``sqrt((pred - target)^2 + eps^2)``.

    A differentiable L1: the ``eps`` floor gives a finite gradient at
    ``pred == target``, where ``|x|`` has none. That is the only reason to
    prefer it -- the *value* is numerically close to L1 at the default, so
    expect no PSNR change from swapping them.

    ``eps`` is **squared inside the root**, spanning both live conventions:
    LapSRN's ``eps=1e-3`` (the default, and the paper that brought Charbonnier
    to SISR) and BasicSR's ``eps=1e-6``.

    Args:
        eps: Intensity-scale floor, squared inside the root. Defaults to
            ``1e-3``.
        reduction: One of ``'mean'``, ``'sum'``, ``'none'``. Defaults to
            ``'mean'``.

    Raises:
        ValueError: If ``reduction`` is not a recognised name.
    """

    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = _check_reduction(reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Charbonnier distance between ``pred`` and ``target``."""
        return _reduce(torch.sqrt((pred - target) ** 2 + self.eps**2), self.reduction)


class TotalVariationLoss(torch.nn.Module):
    """Total-variation loss: isotropic ``sqrt(dx^2 + dy^2 + eps^2)``, reduced per ``reduction``.

    Required, not optional, for SRResNet-VGG22: Ledig et al. Sec. 3.4 add it to
    the VGG22 content loss at weight ``2e-8``, and that is the one VGG variant
    reproducible without a discriminator. Their ``l_TV`` is a norm of the
    gradient *vector*; ``isotropic=False`` gives the common ``|dx| + |dy|``
    reimplementation, a **different function** (up to ``sqrt(2)`` larger on
    diagonals), not a rescaling. Both grids crop to ``(H-1, W-1)`` so the norm
    combines them per pixel.

    Two traps before setting a weight:

    - ``target`` is **ignored** -- a regulariser, not a distance. The uniform
      signature exists so ``WeightedSumLoss`` dispatches every term identically.
    - **The value scales with the intensity range.** The paper's ``2e-8``
      presumes ``[-1, 1]``; under a ``[0, 1]`` processor the same image gives
      exactly half the TV, so the effective weight differs by 2x.

    Args:
        isotropic: Use the paper's ``sqrt(dx^2 + dy^2)`` norm. ``False``
            selects ``|dx| + |dy|``, where ``eps`` is unused. Defaults to
            ``True``.
        eps: Intensity-scale floor, squared inside the root. Non-zero by
            default because ``sqrt`` is non-differentiable at zero and flat
            regions dominate natural images. Defaults to ``1e-4``.
        reduction: One of ``'mean'``, ``'sum'``, ``'none'``. Defaults to
            ``'mean'``.

    Raises:
        ValueError: If ``reduction`` is not a recognised name.
    """

    def __init__(self, isotropic: bool = True, eps: float = 1e-4, reduction: str = "mean"):
        super().__init__()
        self.isotropic = isotropic
        self.eps = eps
        self.reduction = _check_reduction(reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """Total variation of ``pred``. ``target`` is accepted and ignored."""
        dy = pred[..., 1:, :-1] - pred[..., :-1, :-1]
        dx = pred[..., :-1, 1:] - pred[..., :-1, :-1]
        if self.isotropic:
            tv = torch.sqrt(dx * dx + dy * dy + self.eps**2)
        else:
            tv = dx.abs() + dy.abs()
        return _reduce(tv, self.reduction)
