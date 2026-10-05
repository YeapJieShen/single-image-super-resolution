"""RRDBNet -- ESRGAN's generator (Wang et al. 2018), without batch normalisation.

Reference: ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
(https://arxiv.org/abs/1809.00219), Section 3.1 and Figure 4.
"""

from typing import Any, ClassVar, Literal

import torch

from sisr.models.base import SRModel

_SLOPE = 0.2


class _DenseBlock(torch.nn.Module):
    """Five densely connected 3x3 convs; ``x + beta * conv5(...)``.

    ``conv1..4`` each add ``growth`` channels and are followed by a LeakyReLU; every conv
    reads the concatenation of the block input and all earlier outputs. ``conv5`` maps back
    to ``features`` and has no activation.
    """

    def __init__(self, features: int, growth: int, beta: float):
        super().__init__()
        self.beta = beta
        self.conv1 = torch.nn.Conv2d(features, growth, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(features + growth, growth, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(features + 2 * growth, growth, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(features + 3 * growth, growth, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(features + 4 * growth, features, 3, padding=1)
        self.act = torch.nn.LeakyReLU(_SLOPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + self.beta * x5


class _RRDB(torch.nn.Module):
    """Residual-in-Residual Dense Block: three dense blocks, ``x + beta * f(x)``."""

    def __init__(self, features: int, growth: int, beta: float):
        super().__init__()
        self.beta = beta
        self.rdb1 = _DenseBlock(features, growth, beta)
        self.rdb2 = _DenseBlock(features, growth, beta)
        self.rdb3 = _DenseBlock(features, growth, beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.beta * self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNet(SRModel):
    """ESRGAN's generator: RRDB trunk, nearest-neighbour upsampling, no BatchNorm.

    ``head`` conv -> ``num_blocks`` RRDBs -> ``trunk_conv``, added back onto the head
    features; then for each power of two in ``scale`` a **nearest-neighbour x2 resize followed
    by a conv and LeakyReLU** (not a sub-pixel shuffle -- this is the released network, which
    the pretrained weights require); then ``hr_conv`` + LeakyReLU + ``tail``. The output is
    the HR image directly (no bicubic skip). Residual scaling ``beta`` (0.2) multiplies
    both the dense-block and the RRDB residuals.

    Reference:
    - ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks:
      https://arxiv.org/abs/1809.00219

    Args:
        scale: Upscaling factor: a power of two (4 in the paper; the released weights are x4).
        in_out_channels: Input/output channel count (3, RGB).
        features: Width of the trunk (64).
        growth: Channels each dense-block conv adds (32).
        num_blocks: RRDB count (23 in the deep model; 16 in the small one).
        residual_scale: ``beta`` (0.2).

    Raises:
        ValueError: If a size is not a positive int, or ``scale`` is not a power of two >= 2.
    """

    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]] = "native_lr"

    def __init__(
        self,
        scale: int = 4,
        in_out_channels: int = 3,
        features: int = 64,
        growth: int = 32,
        num_blocks: int = 23,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        for name, value in (
            ("scale", scale),
            ("in_out_channels", in_out_channels),
            ("features", features),
            ("growth", growth),
            ("num_blocks", num_blocks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int; got {value!r}.")
        if scale < 2 or scale & (scale - 1):
            raise ValueError(f"scale must be a power of two >= 2; got {scale}.")
        if not 0 < residual_scale <= 1:
            raise ValueError(f"residual_scale must be in (0, 1]; got {residual_scale}.")

        self._hparams = {
            "scale": scale,
            "in_out_channels": in_out_channels,
            "features": features,
            "growth": growth,
            "num_blocks": num_blocks,
            "residual_scale": residual_scale,
        }
        self.head = torch.nn.Conv2d(in_out_channels, features, 3, padding=1)
        self.trunk = torch.nn.Sequential(
            *[_RRDB(features, growth, residual_scale) for _ in range(num_blocks)]
        )
        self.trunk_conv = torch.nn.Conv2d(features, features, 3, padding=1)
        self.up = torch.nn.ModuleList(
            torch.nn.Conv2d(features, features, 3, padding=1) for _ in range(scale.bit_length() - 1)
        )
        self.hr_conv = torch.nn.Conv2d(features, features, 3, padding=1)
        self.tail = torch.nn.Conv2d(features, in_out_channels, 3, padding=1)
        self.act = torch.nn.LeakyReLU(_SLOPE)

    @property
    def variant_tag(self) -> str:
        """Blocks and growth, e.g. ``'23B32G'``."""
        return f"{self._hparams['num_blocks']}B{self._hparams['growth']}G"

    def reset_parameters(self, **kwargs: Any) -> None:
        """The paper's smaller initialisation: He (MSRA) normal, times 0.1.

        Section "Smaller initialization" of the supplement: "multiplying 0.1 for all
        initialization parameters that calculated by MSRA initialization". Fan-in and a ReLU
        gain are assumed (the paper names neither); biases are zero.

        Args:
            **kwargs: Unused; absorbed so callers can invoke this polymorphically.
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
                with torch.no_grad():
                    module.weight.mul_(0.1)
                assert module.bias is not None
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale ``x`` by ``scale``.

        Args:
            x: LR batch, ``(B, in_out_channels, H, W)``.

        Returns:
            ``(B, in_out_channels, H * scale, W * scale)``.
        """
        fea = self.head(x)
        fea = fea + self.trunk_conv(self.trunk(fea))
        for conv in self.up:
            fea = self.act(
                conv(torch.nn.functional.interpolate(fea, scale_factor=2, mode="nearest"))
            )
        return self.tail(self.act(self.hr_conv(fea)))
