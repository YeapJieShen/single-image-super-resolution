import pytest

from sisr.losses.vgg import _parse_layer, _resolve_slice_end

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
