import torch

from sisr.training.self_ensemble import (
    _DIHEDRAL_FLAGS,
    _transform,
    _untransform,
    self_ensemble_forward,
)


def test_untransform_inverts_transform_for_every_dihedral_element():
    """Round trip must hold for all 8 elements, including ones mixing transpose with an odd
    flip — a wrong _untransform ordering (same order as _transform instead of reversed)
    leaves flip-only combos passing (flips commute with each other) but breaks any combo
    where transpose and exactly one flip interact."""
    x = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
    for hflip, vflip, transpose in _DIHEDRAL_FLAGS:
        transformed = _transform(x, hflip, vflip, transpose)
        assert torch.equal(_untransform(transformed, hflip, vflip, transpose), x)


def test_eight_dihedral_transforms_are_pairwise_distinct():
    """Guards against an implementation collapsing two nominally-different elements onto the
    same tensor (e.g. an aliased dim), which would silently self-ensemble over fewer than 8
    real transforms."""
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    seen = {tuple(_transform(x, *flags).flatten().tolist()) for flags in _DIHEDRAL_FLAGS}
    assert len(seen) == 8


def test_self_ensemble_forward_spreads_a_corner_bias_to_all_four_corners():
    """A model_fn that always corrupts the SAME corner of whatever it's given
    (model_fn(t) = t + bias, bias nonzero only at [...,0,0]) has each of the 8 transforms map
    that corner to one of the image's 4 real corners, 2 transforms per corner (D4's
    stabilizer of a corner has order 2) — so the ensembled output must show exactly a
    quarter of the original bias at ALL 4 corners, not the full bias concentrated at 1.
    Exact expected values (not just "improves"), hand-derived from the 8 elements' actual
    corner mapping — a wrong average (wrong divisor, wrong transform set, transform/untransform
    swapped) changes these numbers, not just their sign."""
    x = torch.zeros(1, 1, 4, 4)
    bias = torch.zeros(1, 1, 4, 4)
    bias[..., 0, 0] = 8.0

    def model_fn(t: torch.Tensor) -> torch.Tensor:
        return t + bias

    single = model_fn(x)
    ensembled = self_ensemble_forward(model_fn, x)

    expected_ensembled = torch.zeros(1, 1, 4, 4)
    for r, c in [(0, 0), (3, 0), (0, 3), (3, 3)]:
        expected_ensembled[..., r, c] = 2.0

    torch.testing.assert_close(ensembled - x, expected_ensembled)
    torch.testing.assert_close(single - x, bias)

    mse_single = ((single - x) ** 2).mean()
    mse_ensembled = ((ensembled - x) ** 2).mean()
    assert mse_ensembled < mse_single


def test_self_ensemble_forward_calls_model_fn_exactly_eight_times():
    calls = []

    def model_fn(t: torch.Tensor) -> torch.Tensor:
        calls.append(t.shape)
        return t

    self_ensemble_forward(model_fn, torch.rand(1, 1, 4, 4))
    assert len(calls) == 8


def test_self_ensemble_forward_identity_model_on_asymmetric_input():
    """With an identity model (fn(x) = x), self_ensemble_forward must return the input
    unchanged. Tests that the inverse-transform seam is correct — a mutation that
    re-transforms instead of inverse-transforming (e.g. _transform(f(_transform(x))))
    would fail this test with a 90° or 270° rotation off by 180°. Uses non-square input
    on purpose to exercise the transpose branch."""
    x = torch.arange(15.0).reshape(1, 1, 3, 5)

    def identity_model(t: torch.Tensor) -> torch.Tensor:
        return t

    result = self_ensemble_forward(identity_model, x)
    torch.testing.assert_close(result, x)
