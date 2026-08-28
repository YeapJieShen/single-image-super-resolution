"""SRResNet — the residual generator from Ledig et al. (2017).

Reference: Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network (https://arxiv.org/pdf/1609.04802).
"""

import math
from typing import ClassVar, Literal

import torch

from sisr.models.base import SRModel


class SRResidualBlock(torch.nn.Module):
    """A single residual block: two conv+BN layers with an identity skip connection.

    Args:
        channels: Input/output channel count.
        kernel_size: Kernel size for both conv layers.
        padding: Padding for the conv layers. Defaults to ``'same'``.
    """

    def __init__(self, channels: int, kernel_size: int, padding: str | int = "same"):
        super().__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(channels),
            torch.nn.PReLU(),
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: identity + two conv+BN layers.

        Args:
            x: Input tensor, shape ``(B, channels, H, W)``.

        Returns:
            Output tensor, same shape as *x*.
        """
        identity = x
        x = self.block1(x)
        x = self.block2(x)
        return identity + x


class SRUpsampleBlock(torch.nn.Module):
    """Sub-pixel convolution upsampling block: conv -> PixelShuffle -> PReLU.

    Args:
        channels: Input channel count.
        scale: Upscaling factor. Defaults to ``2``.
        kernel_size: Kernel size for the conv layer. Defaults to ``3``.
    """

    def __init__(self, channels: int, scale: int = 2, kernel_size: int = 3):
        super().__init__()

        self.upsample = torch.nn.Sequential(
            torch.nn.Conv2d(
                channels, channels * (scale**2), kernel_size=kernel_size, padding="same"
            ),
            torch.nn.PixelShuffle(scale),
            torch.nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape ``(B, channels, H, W)``.

        Returns:
            Output tensor with H/W scaled by *scale*.
        """
        return self.upsample(x)


class SRResNet(SRModel):
    """SRResNet: head conv, residual blocks with a skip connection, sub-pixel upsampling, tail conv.

    Reference:
    - Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network: https://arxiv.org/pdf/1609.04802

    Args:
        scale: Upscaling factor. Must be a power of 2.
        in_out_channels: Input/output channel count (e.g. 3 for RGB).
        hidden_channel: Feature channel count used in the residual/upsample blocks.
        kernel_sizes: Kernel sizes for the head, residual, and tail conv layers.
        num_residual_blocks: Number of residual blocks in the network.
        padding: Padding for the conv layers. Defaults to ``'same'``.
    """

    #: Consumes true low-resolution input and upsamples internally (sub-pixel conv).
    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]] = "native_lr"

    def __init__(
        self,
        scale: int,
        in_out_channels: int = 3,
        hidden_channel: int = 64,
        kernel_sizes: tuple[int, ...] = (9, 3, 9),
        num_residual_blocks: int = 16,
        padding: str | int = "same",
    ):
        super().__init__()

        self._check_scale(scale)
        self._check_architecture(kernel_sizes, num_residual_blocks)

        self._hparams = {
            "scale": scale,
            "in_out_channels": in_out_channels,
            "hidden_channel": hidden_channel,
            "kernel_sizes": kernel_sizes,
            "num_residual_blocks": num_residual_blocks,
            "padding": padding,
        }

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_out_channels, hidden_channel, kernel_size=kernel_sizes[0], padding=padding
            ),
            torch.nn.PReLU(),
        )

        self.residual_blocks = torch.nn.Sequential(
            *[
                SRResidualBlock(
                    channels=hidden_channel, kernel_size=kernel_sizes[1], padding=padding
                )
                for _ in range(num_residual_blocks)
            ]
        )

        self.post_residual = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_channel, hidden_channel, kernel_size=kernel_sizes[1], padding=padding
            ),
            torch.nn.BatchNorm2d(hidden_channel),
        )

        self.upsample = torch.nn.Sequential(
            *[
                SRUpsampleBlock(channels=hidden_channel, scale=2)
                for _ in range(int(math.log2(scale)))
            ]
        )

        self.tail = torch.nn.Conv2d(
            hidden_channel, in_out_channels, kernel_size=kernel_sizes[2], padding=padding
        )

    def _check_scale(self, scale: int) -> None:
        """Validates that scale is a positive power of 2."""
        if not isinstance(scale, int) or scale < 1:
            raise ValueError(f"scale must be a positive integer. Got {scale}.")
        if (scale & (scale - 1)) != 0:
            raise ValueError(f"scale must be a power of 2. Got {scale}.")

    def _check_architecture(self, kernel_sizes: tuple[int, ...], num_residual_blocks: int) -> None:
        """Validates kernel_sizes (length-3, positive ints) and num_residual_blocks (positive)."""
        if not isinstance(kernel_sizes, tuple):
            raise ValueError(f"kernel_sizes must be a tuple. Got {type(kernel_sizes)}.")
        if len(kernel_sizes) != 3:
            raise ValueError(
                f"kernel_sizes must have exactly 3 elements for the head, residual, "
                f"and tail layers. Got {kernel_sizes}."
            )
        if any(k < 1 for k in kernel_sizes):
            raise ValueError(
                f"All elements in kernel_sizes must be positive integers. Got {kernel_sizes}."
            )
        if num_residual_blocks < 1:
            raise ValueError(
                f"num_residual_blocks must be a positive integer. Got {num_residual_blocks}."
            )

    @property
    def variant_tag(self) -> str:
        """Blocks and width, e.g. ``'16B64F'`` -- the two knobs that move capacity."""
        return f"{self._hparams['num_residual_blocks']}B{self._hparams['hidden_channel']}F"

    def forward(
        self,
        x: torch.Tensor,
        clamp_output: bool = False,
        clamp_minmax: tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """Forward pass: head -> residual blocks -> upsample -> tail.

        clamp_output is a direct-call convenience for offline inference: the
        SRLightning training and validation paths always call model(x) without
        it, so clamping only takes effect on direct model(x, clamp_output=True)
        calls used to clip the raw output into a displayable range.

        Args:
            x: Input tensor, shape ``(B, in_out_channels, H, W)``.
            clamp_output: Whether to clamp the output to clamp_minmax. Only
                honoured on direct ``model(x, clamp_output=True)`` calls; the
                SRLightning pipeline never sets it.
            clamp_minmax: Bounds for clamping. Defaults to the paper's ``[-1, 1]``
                output range (Ledig et al. §3.2, via ``RGBSignedOutputProcessor``);
                pass ``(0.0, 1.0)`` for weights trained under ``RGBProcessor``.

        Returns:
            Output tensor, shape ``(B, in_out_channels, H*scale, W*scale)``.
        """
        x = self.head(x)
        identity = x
        x = self.residual_blocks(x)
        x = identity + self.post_residual(x)
        x = self.upsample(x)
        x = self.tail(x)

        if clamp_output:
            x = torch.clamp(x, min=clamp_minmax[0], max=clamp_minmax[1])

        return x
