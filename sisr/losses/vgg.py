"""VGG feature-space perceptual losses (Ledig et al. §3.2)."""

from collections.abc import Sequence


def _parse_layer(layer: str) -> tuple[int, int]:
    """Split a ``"vgg<i><j>"`` layer name into its block and conv indices.

    Args:
        layer: Layer name in the paper's ``phi_{i,j}`` shorthand, e.g.
            ``"vgg22"`` for ``phi_{2,2}``.

    Returns:
        ``(i, j)`` — the 1-based block index and the 1-based conv index
        within that block.

    Raises:
        ValueError: If ``layer`` is not ``"vgg"`` followed by exactly two
            digits.
    """
    if len(layer) != 5 or not layer.startswith("vgg") or not layer[3:].isdigit():
        raise ValueError(
            f"layer must have the form 'vgg<i><j>' — two single digits naming "
            f"phi_{{i,j}}, e.g. 'vgg22' or 'vgg54'; got {layer!r}"
        )
    return int(layer[3]), int(layer[4])


def _resolve_slice_end(block_widths: Sequence[int], i: int, j: int, before_activation: bool) -> int:
    """Index one past the last ``features`` layer to keep for ``phi_{i,j}``.

    A torchvision VGG's ``features`` is ``(conv, ReLU) * width`` per block,
    then one ``MaxPool2d``. Ledig et al. define ``phi_{i,j}`` as the feature
    map after the j-th convolution (**after activation**) before the i-th
    maxpool; ESRGAN later argued for the pre-activation map, which is one
    layer shallower.

    Args:
        block_widths: Convolutions per block, e.g. ``(2, 2, 4, 4, 4)`` for
            VGG19.
        i: 1-based block index.
        j: 1-based conv index within block ``i``.
        before_activation: Stop at the conv instead of its ReLU.

    Returns:
        A slice end usable as ``features[:end]``.

    Raises:
        ValueError: If ``i`` is not a block, or ``j`` not a conv of block
            ``i``.
    """
    if not 1 <= i <= len(block_widths):
        raise ValueError(f"block index i must be 1..{len(block_widths)} for this depth; got {i}")
    width = block_widths[i - 1]
    if not 1 <= j <= width:
        deepest = f"vgg{len(block_widths)}{block_widths[-1]}"
        raise ValueError(
            f"conv index j must be 1..{width} — block {i} has {width} convolutions; "
            f"got {j}. The deepest layer at this depth is '{deepest}'."
        )
    offset = sum(2 * w + 1 for w in block_widths[: i - 1])
    conv_index = offset + 2 * (j - 1)
    return conv_index + (1 if before_activation else 2)
