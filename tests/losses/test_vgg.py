import functools

import pytest
import torch

from sisr.losses import VGG16FeatureLoss, VGG19FeatureLoss
from sisr.losses.vgg import _parse_layer, _resolve_slice_end
from sisr.models.srcnn import SRCNN
from sisr.processors import (
    RGBProcessor,
    RGBSignedOutputProcessor,
    YCbCrProcessor,
    YChannelProcessor,
)
from sisr.training import SREvalConfig, SRLightning, SRTrainingConfig

VGG19_WIDTHS = (2, 2, 4, 4, 4)
VGG16_WIDTHS = (2, 2, 3, 3, 3)


def test_parse_layer_splits_the_two_digits():
    assert _parse_layer("vgg22") == (2, 2)
    assert _parse_layer("vgg54") == (5, 4)


@pytest.mark.parametrize("bad", ["vgg2", "vgg222", "conv22", "vggxy", "22", ""])
def test_parse_layer_rejects_anything_but_vgg_plus_two_digits(bad: str):
    with pytest.raises(ValueError, match="vgg"):
        _parse_layer(bad)


@pytest.mark.parametrize(
    ("layer", "expected_conv_idx"),
    [
        ("vgg11", 0),
        ("vgg12", 2),
        ("vgg21", 5),
        ("vgg22", 7),
        ("vgg31", 10),
        ("vgg34", 16),
        ("vgg41", 19),
        ("vgg44", 25),
        ("vgg51", 28),
        ("vgg54", 34),
    ],
)
def test_resolve_slice_end_matches_torchvision_vgg19_indices(layer: str, expected_conv_idx: int):
    """These indices are measured against torchvision's real vgg19.features.
    after-activation keeps the ReLU (conv_idx + 2); before-activation stops at
    the conv itself (conv_idx + 1)."""
    i, j = _parse_layer(layer)

    assert _resolve_slice_end(VGG19_WIDTHS, i, j, before_activation=False) == expected_conv_idx + 2
    assert _resolve_slice_end(VGG19_WIDTHS, i, j, before_activation=True) == expected_conv_idx + 1


def test_resolve_slice_end_matches_torchvision_vgg16_indices():
    """VGG16's blocks 3-5 hold 3 convs, not 4, so indices diverge from VGG19
    past block 2 — conv2_2 is 7 in both, conv5_3 is 28 in VGG16."""
    assert _resolve_slice_end(VGG16_WIDTHS, 2, 2, before_activation=False) == 9
    assert _resolve_slice_end(VGG16_WIDTHS, 5, 3, before_activation=False) == 30


def test_resolve_slice_end_rejects_a_conv_the_block_does_not_have():
    """vgg23 looks plausible and does not exist: block 2 has 2 convs."""
    with pytest.raises(ValueError, match="block 2 has 2"):
        _resolve_slice_end(VGG19_WIDTHS, 2, 3, before_activation=False)


def test_resolve_slice_end_rejects_vgg54_on_vgg16_naming_the_deepest_valid_layer():
    """The one cross-depth trap: vgg54 is SRGAN's headline layer and simply
    does not exist on VGG16."""
    with pytest.raises(ValueError, match="vgg53"):
        _resolve_slice_end(VGG16_WIDTHS, 5, 4, before_activation=False)


def test_resolve_slice_end_rejects_an_out_of_range_block():
    with pytest.raises(ValueError, match="block index"):
        _resolve_slice_end(VGG19_WIDTHS, 6, 1, before_activation=False)


# ---------------------------------------------------------------------------
# VGG*FeatureLoss — every construction passes weights=None to stay offline
# ---------------------------------------------------------------------------


@pytest.fixture
def vgg22() -> VGG19FeatureLoss:
    with pytest.warns(UserWarning, match="randomly initialised"):
        return VGG19FeatureLoss(layer="vgg22", weights=None)


def test_weights_none_warns_that_the_loss_is_meaningless():
    """A random-init VGG computes a perceptual loss over noise. It is the only
    way to keep CI offline, so it must be loud rather than convenient."""
    with pytest.warns(UserWarning, match="randomly initialised"):
        VGG19FeatureLoss(weights=None)


def test_truncates_to_only_the_layers_it_evaluates():
    """vgg22 needs 0.26M of vgg19's 20.02M feature params. Carrying the rest
    would be 80MB of unused weights resident for the whole run."""
    with pytest.warns(UserWarning):
        shallow = VGG19FeatureLoss(layer="vgg22", weights=None)
    with pytest.warns(UserWarning):
        deep = VGG19FeatureLoss(layer="vgg54", weights=None)

    n_shallow = sum(p.numel() for p in shallow._vgg.parameters())
    n_deep = sum(p.numel() for p in deep._vgg.parameters())
    assert n_shallow == pytest.approx(260_000, rel=0.05)
    assert n_deep == pytest.approx(20_024_000, rel=0.05)


def test_vgg16_rejects_vgg54_but_vgg19_accepts_it():
    with pytest.warns(UserWarning):
        VGG19FeatureLoss(layer="vgg54", weights=None)
    with pytest.raises(ValueError, match="vgg53"):
        VGG16FeatureLoss(layer="vgg54", weights=None)


def test_default_layer_constructs_at_both_depths():
    """Guards the one cross-depth footgun in the default arguments."""
    with pytest.warns(UserWarning):
        assert VGG19FeatureLoss(weights=None).layer == "vgg22"
    with pytest.warns(UserWarning):
        assert VGG16FeatureLoss(weights=None).layer == "vgg22"


def test_bind_makes_the_two_rgb_ranges_produce_identical_features(vgg22):
    """THE test for the whole bind design: a [0, 1] tensor under RGBProcessor
    and the same image as 2x-1 under RGBSignedOutputProcessor are the same
    picture, so they must score identically. Without bind reading
    output_range, the signed path feeds [-1, 1] to VGG and is silently wrong."""
    x = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)

    vgg22.bind(RGBProcessor())
    unsigned = vgg22(x, target)

    vgg22.bind(RGBSignedOutputProcessor())
    signed = vgg22(x * 2 - 1, target * 2 - 1)

    assert signed.item() == pytest.approx(unsigned.item(), rel=1e-5)


def test_bind_rejects_a_one_channel_processor_and_names_the_opt_in(vgg22):
    with pytest.raises(ValueError, match="grayscale_to_rgb"):
        vgg22.bind(YChannelProcessor())


def test_grayscale_to_rgb_opts_in_to_the_one_channel_case():
    with pytest.warns(UserWarning):
        loss = VGG19FeatureLoss(layer="vgg22", grayscale_to_rgb=True, weights=None)
    loss.bind(YChannelProcessor())

    got = loss(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32))

    assert got.ndim == 0 and torch.isfinite(got)


def test_bind_rejects_a_non_rgb_three_channel_processor_and_names_the_opt_in(vgg22):
    """YCbCrProcessor emits 3 channels — the shape check alone would accept it, but
    its planes are not R/G/B, so ImageNet normalisation and every downstream VGG
    feature are meaningless. Must name allow_non_rgb, the escape hatch."""
    with pytest.raises(ValueError, match="allow_non_rgb"):
        vgg22.bind(YCbCrProcessor())


def test_allow_non_rgb_opts_in_to_the_ycbcr_case():
    with pytest.warns(UserWarning):
        loss = VGG19FeatureLoss(layer="vgg22", allow_non_rgb=True, weights=None)
    loss.bind(YCbCrProcessor())

    got = loss(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))

    assert got.ndim == 0 and torch.isfinite(got)


def test_error_message_names_the_concrete_subclass(vgg22):
    with pytest.raises(ValueError, match="VGG19FeatureLoss"):
        vgg22.bind(YChannelProcessor())


def test_feature_scale_squares_into_an_mse_and_is_linear_in_an_l1():
    """feature_scale multiplies the feature maps (the paper's wording), so an
    MSE of them carries its square. Switching distance changes the magnitude
    by 12.75x, which is why the docstring has to say so."""
    x, target = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    for distance, power in (("mse", 2), ("l1", 1)):
        with pytest.warns(UserWarning):
            unscaled = VGG19FeatureLoss(
                layer="vgg22", feature_scale=1.0, distance=distance, weights=None
            )
        with pytest.warns(UserWarning):
            scaled = VGG19FeatureLoss(
                layer="vgg22", feature_scale=0.5, distance=distance, weights=None
            )
        scaled._vgg.load_state_dict(unscaled._vgg.state_dict())
        # An unbound loss now refuses to compute; both sides get the same
        # binding so the ratio still isolates feature_scale.
        unscaled.bind(RGBProcessor())
        scaled.bind(RGBProcessor())

        ratio = scaled(x, target).item() / unscaled(x, target).item()

        assert ratio == pytest.approx(0.5**power, rel=1e-4), distance


def test_before_activation_changes_the_features(vgg22):
    """ESRGAN's variant is a real behaviour change, not a documentation note."""
    with pytest.warns(UserWarning):
        pre = VGG19FeatureLoss(layer="vgg22", before_activation=True, weights=None)
    pre._vgg.load_state_dict(vgg22._vgg.state_dict(), strict=True)
    x, target = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    # Same binding on both, so the difference isolates before_activation.
    pre.bind(RGBProcessor())
    vgg22.bind(RGBProcessor())

    assert pre(x, target).item() != pytest.approx(vgg22(x, target).item())


def test_describe_omits_default_knobs(vgg22):
    assert vgg22.describe() == "VGG19FeatureLoss(vgg22)"


def test_describe_appends_non_default_knobs():
    """before_activation and distance both change the loss materially, so a
    non-default value must survive into the only recipe string that lands in
    checkpoint metadata and HParams."""
    with pytest.warns(UserWarning):
        loss = VGG19FeatureLoss(layer="vgg22", before_activation=True, distance="l1", weights=None)

    assert loss.describe() == "VGG19FeatureLoss(vgg22, before_activation=True, distance=l1)"


def test_rejects_an_unknown_distance():
    with pytest.raises(ValueError, match="distance"):
        VGG19FeatureLoss(distance="huber", weights=None)


# --- the grad contract: three independent facts ---


def test_frozen_vgg_takes_no_gradient_but_still_passes_one_through(vgg22):
    """Frozen weights and a grad-carrying forward are independent. Wrapping the
    SR branch in no_grad would kill all learning while the loss still moved."""
    vgg22.bind(RGBProcessor())
    model = torch.nn.Conv2d(3, 3, 3, padding=1)
    pred = model(torch.rand(1, 3, 32, 32))

    vgg22(pred, torch.rand(1, 3, 32, 32)).backward()

    assert model.weight.grad is not None, "gradient must reach the generator"
    assert all(p.grad is None for p in vgg22._vgg.parameters()), "VGG must not accumulate"
    assert not any(p.requires_grad for p in vgg22._vgg.parameters())


def test_target_branch_is_not_differentiated(vgg22):
    """The target is a constant; building its graph is wasted memory."""
    vgg22.bind(RGBProcessor())
    target = torch.rand(1, 3, 32, 32, requires_grad=True)

    vgg22(torch.rand(1, 3, 32, 32, requires_grad=True), target).backward()

    assert target.grad is None


def test_vgg_params_are_invisible_to_the_optimizer(vgg22):
    """A frozen 20M-param VGG inside .parameters() is an accident waiting for
    someone to call requires_grad_(True) on the module."""
    lit = SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(crop_border=0),
        criterion=vgg22,
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )

    assert not any(p.numel() > 100_000 for p in lit.parameters())


# --- checkpoint hygiene ---


def test_vgg_is_absent_from_the_modules_state_dict(vgg22):
    """20M frozen params would be ~80MB in every .ckpt, and the template writes
    two monitors at top-k 3. It would also break strict-mode loading of a
    checkpoint trained under a different criterion."""
    lit = SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
        eval_config=SREvalConfig(crop_border=0),
        criterion=vgg22,
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )

    keys = list(lit.state_dict())

    assert not any("criterion" in k for k in keys), keys
    assert all(k.startswith("model.") for k in keys), keys


def test_an_mse_checkpoint_loads_strictly_into_a_vgg_configured_module(vgg22):
    """The practical payoff of keeping VGG out of state_dict: recipes stay
    interchangeable across a resume."""
    args = dict(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    mse_state = SRLightning(**args).state_dict()

    SRLightning(**args, criterion=vgg22).load_state_dict(mse_state, strict=True)


def test_to_moves_the_unregistered_vgg_with_the_module(vgg22):
    """VGG lives outside the module tree, so .to() reaches it only through the
    _apply override. Without it, the first real step dies on a device mismatch."""
    vgg22.to(torch.float64)

    assert next(vgg22._vgg.parameters()).dtype is torch.float64


def test_to_empty_with_recurse_false_leaves_the_vgg_weights_untouched(vgg22):
    """_apply must forward recurse rather than always recursing: to_empty's
    recurse=False call is meant to touch only this module's own tensors (there
    are none outside _vgg), and ignoring it would silently replace every VGG
    weight with torch.empty_like's uninitialised memory."""
    before = [p.clone() for p in vgg22._vgg.parameters()]

    vgg22.to_empty(device="cpu", recurse=False)

    after = list(vgg22._vgg.parameters())
    assert len(after) == len(before)
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a)


# ---------------------------------------------------------------------------
# "never bound" must be loud, not silently [0, 1]
# ---------------------------------------------------------------------------


def test_unbound_vgg_loss_refuses_to_compute(vgg22: VGG19FeatureLoss):
    """__init__ set _gain=1.0, _offset=0.0 -- the identity -- so an unbound loss
    silently behaved as though the model emits [0, 1], and "never bound" was
    indistinguishable at runtime from "bound to a [0, 1] processor".

    The identical question is already answered the other way one seam over:
    SRProcessor.output_range is abstract rather than defaulted, and says why --
    "a wrong inherited default is the failure this exists to prevent". The
    consumer of that carefully-abstract value must not hand itself a default."""
    with pytest.raises(RuntimeError, match="bind"):
        vgg22(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))


def test_unbound_error_names_both_live_paths(vgg22: VGG19FeatureLoss):
    """Whoever hits this has written a custom composite and needs to know the
    contract exists, so the message has to say what to do about it."""
    with pytest.raises(RuntimeError) as excinfo:
        vgg22(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    message = str(excinfo.value)
    assert "WeightedSumLoss" in message
    assert "bind" in message


def test_bound_directly_is_unaffected(vgg22: VGG19FeatureLoss):
    """Shipped path 1: an SRLoss handed to SRLightning as the criterion."""
    vgg22.bind(RGBProcessor())
    assert torch.isfinite(vgg22(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)))


def test_bound_through_weighted_sum_is_unaffected(vgg22: VGG19FeatureLoss):
    """Shipped path 2: nested in WeightedSumLoss, which forwards bind() to its
    SRLoss terms. This is how the flagship reproduction's content loss is wired."""
    from sisr.losses import WeightedSumLoss

    composite = WeightedSumLoss(terms={"vgg": vgg22}, weights={"vgg": 1.0})
    composite.bind(RGBSignedOutputProcessor())
    assert torch.isfinite(composite(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)))


def test_signed_range_processor_actually_changes_the_mapping(vgg22: VGG19FeatureLoss):
    """The bug's consequence, pinned: [-1, 1] and [0, 1] must not agree. If they
    did, the silent default would have been harmless and this is all theatre."""
    x = torch.rand(1, 3, 16, 16) * 2 - 1
    y = torch.rand(1, 3, 16, 16) * 2 - 1
    vgg22.bind(RGBProcessor())
    as_unit = vgg22(x, y)
    vgg22.bind(RGBSignedOutputProcessor())
    as_signed = vgg22(x, y)
    assert not torch.isclose(as_unit, as_signed), (
        "a [-1, 1] model scored through [0, 1] logic must differ -- otherwise "
        "the silently-wrong default cost nothing"
    )
