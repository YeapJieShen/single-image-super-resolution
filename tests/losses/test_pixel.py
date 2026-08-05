import math

import pytest
import torch

from sisr.losses import CharbonnierLoss, TotalVariationLoss

# ---------------------------------------------------------------------------
# CharbonnierLoss
# ---------------------------------------------------------------------------


def test_charbonnier_matches_the_closed_form():
    pred = torch.tensor([[[[0.0, 3.0]]]])
    target = torch.tensor([[[[0.0, 0.0]]]])
    eps = 0.5

    got = CharbonnierLoss(eps=eps)(pred, target)

    expected = (math.sqrt(0.0 + eps**2) + math.sqrt(9.0 + eps**2)) / 2
    assert got.item() == pytest.approx(expected)


def test_charbonnier_eps_is_squared_so_1e_6_reproduces_basicsr():
    """BasicSR computes sqrt(diff**2 + 1e-12). One formula, both conventions —
    if eps stopped being squared, this must fail. float64 and a tolerance below
    the 1.25e-7 relative effect size are both required: in float32 the two
    variants differ by less than the dtype's own epsilon."""
    pred = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    target = torch.zeros_like(pred)

    got = CharbonnierLoss(eps=1e-6)(pred, target)

    assert got.item() == pytest.approx(math.sqrt(4.0 + 1e-12), rel=1e-9)


def test_charbonnier_gradient_is_finite_where_l1_is_not():
    """The entire reason to prefer Charbonnier over L1: |x| has an undefined
    derivative at 0, and a pixel-perfect prediction is the common case."""
    pred = torch.zeros(1, 1, 4, 4, requires_grad=True)

    CharbonnierLoss()(pred, torch.zeros(1, 1, 4, 4)).backward()

    assert torch.isfinite(pred.grad).all()
    assert torch.equal(pred.grad, torch.zeros_like(pred.grad))


@pytest.mark.parametrize(
    ("reduction", "shape"), [("mean", ()), ("sum", ()), ("none", (1, 1, 2, 2))]
)
def test_charbonnier_reduction_modes(reduction: str, shape: tuple[int, ...]):
    pred, target = torch.rand(1, 1, 2, 2), torch.rand(1, 1, 2, 2)

    got = CharbonnierLoss(reduction=reduction)(pred, target)

    assert got.shape == shape


def test_charbonnier_rejects_an_unknown_reduction():
    with pytest.raises(ValueError, match="reduction"):
        CharbonnierLoss(reduction="average")


# ---------------------------------------------------------------------------
# TotalVariationLoss
# ---------------------------------------------------------------------------


def _ramp(direction: str, n: int = 8) -> torch.Tensor:
    """Unit-slope ramp on an n x n grid, as a (1, 1, n, n) batch."""
    yy, xx = torch.meshgrid(torch.arange(float(n)), torch.arange(float(n)), indexing="ij")
    field = {"x": xx, "y": yy, "diag": (xx + yy) / 2}[direction]
    return (field / n)[None, None]


def test_tv_isotropic_equals_anisotropic_on_an_axis_aligned_ramp():
    """Only one derivative is nonzero, so sqrt(dx^2 + dy^2) == |dx| + |dy|."""
    ramp = _ramp("x")

    iso = TotalVariationLoss(isotropic=True)(ramp)
    aniso = TotalVariationLoss(isotropic=False)(ramp)

    assert iso.item() == pytest.approx(aniso.item(), abs=1e-6)


def test_tv_anisotropic_is_sqrt2_times_isotropic_on_a_45_degree_ramp():
    """The measurement that proves the two are different functions, not two
    scalings of one: dx == dy makes |dx| + |dy| exactly sqrt(2) * ||(dx, dy)||.
    A step edge cannot show this (one nonzero derivative per pixel)."""
    diag = _ramp("diag")

    iso = TotalVariationLoss(isotropic=True)(diag)
    aniso = TotalVariationLoss(isotropic=False)(diag)

    assert aniso.item() / iso.item() == pytest.approx(math.sqrt(2), rel=1e-4)


def test_tv_default_eps_keeps_a_flat_patch_differentiable():
    """sqrt(0) has an infinite derivative, and flat regions dominate natural
    images — eps=0 NaNs on the first flat crop of a real run."""
    flat = torch.zeros(1, 1, 8, 8, requires_grad=True)
    TotalVariationLoss()(flat).backward()
    assert torch.isfinite(flat.grad).all()

    unguarded = torch.zeros(1, 1, 8, 8, requires_grad=True)
    TotalVariationLoss(eps=0.0)(unguarded).backward()
    assert not torch.isfinite(unguarded.grad).all()


def test_tv_ignores_its_target_argument():
    """TV is a regulariser, not a distance. The uniform (pred, target)
    signature is what lets WeightedSumLoss dispatch every term identically."""
    pred = torch.rand(1, 3, 8, 8)
    tv = TotalVariationLoss()

    assert tv(pred, torch.rand(1, 3, 8, 8)).item() == pytest.approx(tv(pred).item())


def test_tv_reduces_over_a_common_grid_one_smaller_in_both_dims():
    """The isotropic norm has to combine two differently-shaped difference
    grids, so both are cropped to (H-1, W-1). 'none' exposes that shape."""
    got = TotalVariationLoss(reduction="none")(torch.rand(2, 3, 8, 6))

    assert got.shape == (2, 3, 7, 5)


def test_tv_is_range_dependent_by_exactly_the_range_ratio():
    """Documents why the paper's 2e-8 weight presumes a [-1, 1] output range:
    the same image mapped to [-1, 1] doubles the TV value, so under
    RGBProcessor the effective weight differs 2x."""
    x01 = torch.rand(1, 3, 16, 16)
    tv = TotalVariationLoss()

    assert tv(x01 * 2 - 1).item() == pytest.approx(2 * tv(x01).item(), rel=1e-5)
