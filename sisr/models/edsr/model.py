"""EDSR -- the enhanced deep super-resolution network of Lim et al. (2017).

Reference: Enhanced Deep Residual Networks for Single Image Super-Resolution
(https://arxiv.org/abs/1707.02921).
"""

import math
from typing import Any, ClassVar, Literal

import torch

from sisr.models.base import SRModel

#: The released models' training-set RGB mean (DIV2K), in [0, 1].
DIV2K_RGB_MEAN = (0.4488, 0.4371, 0.4040)


class _ResBlock(torch.nn.Module):
    """``x + residual_scale * conv(relu(conv(x)))`` -- no BatchNorm."""

    def __init__(self, features: int, residual_scale: float):
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1 = torch.nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.conv2(torch.relu(self.conv1(x)))


class EDSR(SRModel):
    """Residual trunk without BatchNorm, mean shift, and a sub-pixel upsampler.

    ``head`` conv -> ``num_blocks`` residual blocks -> one more conv, added back onto the head
    features -> an upsampler -> ``tail`` conv. The upsampler is one ``conv`` to ``4 * features``
    and a ``PixelShuffle(2)`` per factor of two, or one ``conv`` to ``9 * features`` and a
    ``PixelShuffle(3)`` for x3 (so x2 and x3 have one stage, x4 two).

    **The model takes and returns [0, 1].** The released weights were trained on the 0-255
    range with the dataset mean subtracted at the input and added back at the output; both are
    done inside ``forward`` (``rgb_range`` scales, ``rgb_mean`` shifts) as fixed buffers, not
    trainable 1x1 convs as in the authors' code, so the model stays an ordinary
    [0, 1] -> [0, 1] :class:`~sisr.models.base.SRModel`. The port verifies the checkpoint's
    mean-shift convs equal these buffers and drops them.

    Presets from the paper (Table 1): **baseline** ``num_blocks=16, features=64,
    residual_scale=1`` and **EDSR** ``num_blocks=32, features=256, residual_scale=0.1`` (the
    scaling 0.1 stabilises the wide model). EDSR+ is this model with geometric self-ensemble at
    test time -- ``eval_config.self_ensemble: true``.

    Reference:
    - Enhanced Deep Residual Networks for Single Image Super-Resolution:
      https://arxiv.org/abs/1707.02921

    Args:
        scale: Upscaling factor: 2, 3 or another power of two.
        in_out_channels: Input/output channel count (3, RGB; the mean shift is RGB).
        features: Trunk width.
        num_blocks: Residual block count.
        residual_scale: Factor on each block's residual.
        rgb_mean: Per-channel mean in [0, 1].
        rgb_range: The range the weights were trained at (255).

    Raises:
        ValueError: If a size is not a positive int, ``scale`` is neither 3 nor a power of two,
            or ``rgb_mean`` does not have ``in_out_channels`` entries.
    """

    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]] = "native_lr"
    mean: torch.Tensor

    def __init__(
        self,
        scale: int,
        in_out_channels: int = 3,
        features: int = 64,
        num_blocks: int = 16,
        residual_scale: float = 1.0,
        rgb_mean: tuple[float, ...] = DIV2K_RGB_MEAN,
        rgb_range: float = 255.0,
    ):
        super().__init__()
        for name, value in (
            ("scale", scale),
            ("in_out_channels", in_out_channels),
            ("features", features),
            ("num_blocks", num_blocks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int; got {value!r}.")
        if scale != 3 and (scale < 2 or scale & (scale - 1)):
            raise ValueError(f"scale must be 3 or a power of two >= 2; got {scale}.")
        if len(rgb_mean) != in_out_channels:
            raise ValueError(f"rgb_mean needs {in_out_channels} entries; got {tuple(rgb_mean)}.")
        if rgb_range <= 0:
            raise ValueError(f"rgb_range must be positive; got {rgb_range}.")

        self._hparams = {
            "scale": scale,
            "in_out_channels": in_out_channels,
            "features": features,
            "num_blocks": num_blocks,
            "residual_scale": residual_scale,
            "rgb_mean": list(rgb_mean),
            "rgb_range": rgb_range,
        }
        self.rgb_range = rgb_range
        self.register_buffer(
            "mean", torch.tensor(rgb_mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False
        )
        self.head = torch.nn.Conv2d(in_out_channels, features, 3, padding=1)
        self.blocks = torch.nn.Sequential(
            *[_ResBlock(features, residual_scale) for _ in range(num_blocks)]
        )
        self.body_conv = torch.nn.Conv2d(features, features, 3, padding=1)
        factors = [3] if scale == 3 else [2] * int(math.log2(scale))
        self.up = torch.nn.ModuleList(
            torch.nn.Conv2d(features, features * f * f, 3, padding=1) for f in factors
        )
        self._factors = factors
        self.tail = torch.nn.Conv2d(features, in_out_channels, 3, padding=1)

    @property
    def variant_tag(self) -> str:
        """Blocks and width, e.g. ``'16B64F'`` or ``'32B256F'``."""
        return f"{self._hparams['num_blocks']}B{self._hparams['features']}F"

    def reset_parameters(self, **kwargs: Any) -> None:
        """No-op: the paper states no initialisation, so PyTorch's default stands.

        Args:
            **kwargs: Unused; absorbed so callers can invoke this polymorphically.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale ``x`` by ``scale``.

        Args:
            x: LR batch in [0, 1], ``(B, in_out_channels, H, W)``.

        Returns:
            ``(B, in_out_channels, H * scale, W * scale)`` in [0, 1] (unclamped).
        """
        h = self.head((x - self.mean) * self.rgb_range)
        h = h + self.body_conv(self.blocks(h))
        for conv, factor in zip(self.up, self._factors, strict=True):
            h = torch.nn.functional.pixel_shuffle(conv(h), factor)
        return self.tail(h) / self.rgb_range + self.mean
