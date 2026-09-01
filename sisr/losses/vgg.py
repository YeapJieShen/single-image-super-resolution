"""VGG feature-space perceptual losses (Ledig et al. §3.2)."""

import math
import warnings
from collections.abc import Callable, Sequence
from typing import ClassVar, Self

import torch
import torchvision

from ..processors import SRProcessor
from .base import SRLoss


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


#: ImageNet statistics torchvision's VGG weights were trained under.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_DISTANCES = {"mse": torch.nn.functional.mse_loss, "l1": torch.nn.functional.l1_loss}


class _VGGFeatureLoss(SRLoss):
    """Shared implementation of a VGG feature-space loss. See the subclasses.

    Args:
        layer: Feature layer in the paper's ``phi_{i,j}`` shorthand, e.g.
            ``"vgg22"``. Defaults to ``"vgg22"`` — valid at every depth, and
            the layer of the reproducible SRResNet-VGG22 recipe.
        before_activation: Take the pre-ReLU feature map (ESRGAN) instead of
            the post-ReLU one (SRGAN). Defaults to ``False``.
        feature_scale: Multiplies the **feature maps**, per Ledig et al.'s
            footnote, so an MSE of them carries its square: the default
            ``1/12.75`` yields the paper's ``0.006``. Under
            ``distance='l1'`` the effect is linear instead, i.e. switching
            distance changes the loss magnitude by ``12.75x``. Defaults to
            ``1 / 12.75``.
        input_norm: Apply ImageNet mean/std normalisation, which is what the
            weights were trained under. Defaults to ``True``.
        distance: ``'mse'`` (Ledig eq. 5) or ``'l1'`` (the BasicSR/ESRGAN
            habit). Defaults to ``'mse'``.
        grayscale_to_rgb: Replicate a 1-channel model output across RGB
            instead of refusing it. Off by default because a Y-channel VGG
            loss is nobody's published recipe. Defaults to ``False``.
        allow_non_rgb: Skip the RGB-colorspace check for a 3-channel
            processor, so e.g. :class:`~sisr.processors.ycbcr.YCbCrProcessor`
            can be paired with this loss anyway. Off by default because
            feeding Y/Cb/Cr to VGG as though it were R/G/B is not a published
            recipe. Defaults to ``False``.
        weights: A torchvision weights-enum name, or ``None`` for a random
            initialisation — which makes the loss meaningless and warns, and
            exists so tests can stay offline. Defaults to
            ``"IMAGENET1K_V1"``.

    Raises:
        ValueError: If ``distance`` is unknown, or ``layer`` names a
            convolution this depth does not have.
    """

    #: Convolutions per block, which is what makes a layer name valid or not.
    BLOCK_WIDTHS: ClassVar[tuple[int, ...]]
    #: Name of the ``torchvision.models`` builder for this depth.
    MODEL_NAME: ClassVar[str]

    # Declared here so attribute access resolves to these types instead of
    # falling back to nn.Module.__getattr__'s `Tensor | Module` union: `_vgg`
    # is assigned via object.__setattr__ (see __init__) and `_mean`/`_std` via
    # register_buffer, neither of which mypy can see as a normal `self.x = ...`.
    _vgg: torch.nn.Module
    _mean: torch.Tensor
    _std: torch.Tensor

    def __init__(
        self,
        layer: str = "vgg22",
        before_activation: bool = False,
        feature_scale: float = 1 / 12.75,
        input_norm: bool = True,
        distance: str = "mse",
        grayscale_to_rgb: bool = False,
        allow_non_rgb: bool = False,
        weights: str | None = "IMAGENET1K_V1",
    ):
        super().__init__()
        if distance not in _DISTANCES:
            raise ValueError(f"distance must be one of {tuple(_DISTANCES)}; got {distance!r}")
        i, j = _parse_layer(layer)
        slice_end = _resolve_slice_end(self.BLOCK_WIDTHS, i, j, before_activation)

        if weights is None:
            warnings.warn(
                f"{type(self).__name__}(weights=None) builds a randomly initialised "
                f"VGG, so the perceptual loss it computes is meaningless. This exists "
                f"for offline tests — do not train with it.",
                UserWarning,
                stacklevel=2,
            )
        features = getattr(torchvision.models, self.MODEL_NAME)(weights=weights).features
        vgg = torch.nn.Sequential(*list(features)[:slice_end]).eval()
        vgg.requires_grad_(False)
        # object.__setattr__ keeps this out of _modules: nn.Module.__setattr__
        # would register it, putting up to 20M frozen params into every
        # state_dict (so every .ckpt) and making a checkpoint non-loadable in
        # strict mode against a module built with a different criterion. _apply
        # below is what still carries .to()/dtype changes across.
        object.__setattr__(self, "_vgg", vgg)

        self.layer = layer
        self.before_activation = before_activation
        self.feature_scale = feature_scale
        self.input_norm = input_norm
        self.distance = distance
        self.grayscale_to_rgb = grayscale_to_rgb
        self.allow_non_rgb = allow_non_rgb
        # Sentinel, NOT the identity. An identity default would make "never
        # bound" and "bound to a [0, 1] processor" indistinguishable at
        # runtime, and the wrong one of those trains on features that mean
        # something other than the recipe says. SRProcessor.output_range --
        # the value bind() reads -- is abstract rather than defaulted for
        # exactly this reason, and its consumer must not undo that by handing
        # itself a fallback.
        self._gain: float | None = None
        self._offset: float | None = None
        # persistent=False: constants, and a distributable checkpoint should not
        # carry them.
        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        """Carry ``.to()`` / dtype changes to the unregistered VGG."""
        applied = super()._apply(fn, recurse)
        if recurse:
            self._vgg._apply(fn)
        return applied

    def bind(self, processor: SRProcessor) -> None:
        """Derive the affine map from the model's output range into ``[0, 1]``.

        Until this runs the loss refuses to compute -- see :meth:`_features`.

        Raises:
            ValueError: If the processor emits a channel count or colorspace
                this loss has no published recipe for, or a non-increasing
                ``output_range``.
        """
        channels = processor.model_channels
        if channels != 3 and not self.grayscale_to_rgb:
            raise ValueError(
                f"{type(self).__name__} needs 3-channel RGB but "
                f"{type(processor).__name__} emits {channels} channel(s). VGG "
                f"perceptual loss is not a published Y-channel recipe. Set "
                f"grayscale_to_rgb: true to replicate the single channel across "
                f"RGB anyway, or pair the model with an RGB processor."
            )
        if channels == 3 and processor.output_colorspace != "RGB" and not self.allow_non_rgb:
            raise ValueError(
                f"{type(self).__name__} normalises with RGB ImageNet statistics, but "
                f"{type(processor).__name__} emits {processor.output_colorspace} planes. "
                f"Feeding Y/Cb/Cr to VGG as though they were R/G/B is not a published "
                f"recipe and trains on features that mean nothing. Pair the model with "
                f"an RGB processor, or set allow_non_rgb: true to experiment anyway."
            )
        low, high = processor.output_range
        if high <= low:
            raise ValueError(
                f"{type(processor).__name__}.output_range must be increasing; got {(low, high)}"
            )
        self._gain = 1.0 / (high - low)
        self._offset = -low / (high - low)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Map into ``[0, 1]``, normalise, and take the truncated VGG's output.

        Raises:
            RuntimeError: If :meth:`bind` has not run, so the model's output
                range is unknown. Both shipped paths bind, so this is reachable
                only through a container that is not itself an
                :class:`~sisr.losses.base.SRLoss`.
        """
        if self._gain is None or self._offset is None:
            raise RuntimeError(
                f"{type(self).__name__} was never bound to a processor, so it does not know "
                "what range the model emits. Reconstructing [-1, 1] output with [0, 1] logic "
                "feeds ImageNet normalisation data it was not designed for: training proceeds, "
                "the number looks plausible, and the features mean something other than the "
                "recipe says. Either put this term inside a WeightedSumLoss (which forwards "
                "bind() to its SRLoss terms) or hand it to SRLightning as the criterion "
                "(which binds it directly). A plain nn.ModuleList or a custom composite does "
                "neither -- call bind(processor) yourself if you need one."
            )
        x = x * self._gain + self._offset
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if self.input_norm:
            x = (x - self._mean) / self._std
        return self._vgg(x) * self.feature_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Distance between the VGG features of ``pred`` and of ``target``."""
        pred_features = self._features(pred)
        # The target is a constant, so its graph is pure waste. pred's is not:
        # it is the only path a gradient has back to the generator.
        with torch.no_grad():
            target_features = self._features(target)
        return _DISTANCES[self.distance](pred_features, target_features)

    def describe(self) -> str:
        """e.g. ``"VGG19FeatureLoss(vgg22)"``, or with non-default knobs appended.

        ``before_activation``, ``distance``, ``feature_scale`` and
        ``input_norm`` all change the loss materially (see the class
        docstring), yet this string is the only record of the criterion in
        checkpoint metadata, HParams and provenance — so any of them that
        differs from its default is appended, e.g.
        ``"VGG19FeatureLoss(vgg22, before_activation=True, distance=l1)"``.
        """
        extras = []
        if self.before_activation:
            extras.append(f"before_activation={self.before_activation}")
        if self.distance != "mse":
            extras.append(f"distance={self.distance}")
        if not math.isclose(self.feature_scale, 1 / 12.75, rel_tol=1e-9):
            extras.append(f"feature_scale={self.feature_scale:g}")
        if not self.input_norm:
            extras.append(f"input_norm={self.input_norm}")
        suffix = "".join(f", {extra}" for extra in extras)
        return f"{type(self).__name__}({self.layer}{suffix})"


class VGG19FeatureLoss(_VGGFeatureLoss):
    """VGG19 feature-space loss — the depth Ledig et al. specify.

    ``phi_{i,j}`` is the feature map after the j-th convolution before the
    i-th maxpool; ``"vgg22"`` and ``"vgg54"`` are the paper's two variants.
    Constructor arguments are documented on :class:`_VGGFeatureLoss`.
    """

    BLOCK_WIDTHS: ClassVar[tuple[int, ...]] = (2, 2, 4, 4, 4)
    MODEL_NAME: ClassVar[str] = "vgg19"


class VGG16FeatureLoss(_VGGFeatureLoss):
    """VGG16 feature-space loss — offered for experimentation, not fidelity.

    Blocks 3-5 hold three convolutions rather than four, so the deepest layer
    is ``"vgg53"`` and ``"vgg54"`` raises. Constructor arguments are
    documented on :class:`_VGGFeatureLoss`.
    """

    BLOCK_WIDTHS: ClassVar[tuple[int, ...]] = (2, 2, 3, 3, 3)
    MODEL_NAME: ClassVar[str] = "vgg16"
