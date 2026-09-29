"""ESPCN -- efficient sub-pixel convolutional network from Shi et al. (2016).

Reference: Real-Time Single Image and Video Super-Resolution Using an Efficient
Sub-Pixel Convolutional Neural Network (https://arxiv.org/pdf/1609.05158).

**Not yet reproduced.** The paper builds its LR images by Gaussian blur of an
unstated sigma followed by subsampling, and no official weights exist to score
against. This project degrades with its standard bicubic MATLAB ``imresize``
(:mod:`sisr.datasets.native_lr`) instead, so a figure from this model is
comparable to the other native-LR models here and not to the paper's tables.
"""

from collections.abc import Sequence
from typing import ClassVar, Literal

import torch

from sisr.models.base import SRModel, as_int_tuple


class ESPCN(SRModel):
    """Convolutions on the LR grid, then one sub-pixel shuffle to the HR grid.

    ``Conv(c, f1, k1) -> Tanh -> Conv(f1, f2, k2) -> Tanh -> Conv(f2, c*scale**2, k3)
    -> PixelShuffle(scale)``. The paper chose tanh over ReLU on measured results, and
    the output is linear (no activation after the shuffle).

    Args:
        scale: Upscaling factor, a positive integer (the paper reports x3 and x4).
        in_out_channels: Input/output channel count (1 for the paper's Y channel).
        filters: Feature counts of the hidden layers. Defaults to ``(64, 32)``.
        kernel_sizes: One odd kernel size per convolution, hidden layers then the
            last, so ``len(filters) + 1`` entries. Defaults to ``(5, 3, 3)``.
        padding: ``'same'``, or the int ``(k - 1) // 2`` for every kernel. The
            shuffle multiplies any size a convolution loses by ``scale``, so the
            convolutions must preserve their spatial size.
    """

    #: Consumes true low-resolution input and upsamples internally (sub-pixel conv).
    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]] = "native_lr"

    def __init__(
        self,
        scale: int,
        in_out_channels: int = 1,
        filters: Sequence[int] = (64, 32),
        kernel_sizes: Sequence[int] = (5, 3, 3),
        padding: str | int = "same",
    ):
        super().__init__()

        filters = as_int_tuple("filters", filters)
        kernel_sizes = as_int_tuple("kernel_sizes", kernel_sizes)
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError(f"scale must be a positive integer. Got {scale!r}.")
        if len(kernel_sizes) != len(filters) + 1:
            raise ValueError(
                f"kernel_sizes must have len(filters) + 1 = {len(filters) + 1} elements "
                f"(one per convolution). Got {kernel_sizes}."
            )
        if any(k < 1 for k in (*filters, *kernel_sizes)):
            raise ValueError(
                f"All elements of filters and kernel_sizes must be positive integers. "
                f"Got filters={filters}, kernel_sizes={kernel_sizes}."
            )
        if any(k % 2 == 0 for k in kernel_sizes):
            raise ValueError(
                f"kernel_sizes must all be odd, so the padding keeps each convolution "
                f"shape-preserving; an even kernel shifts the output grid. Got {kernel_sizes}."
            )
        if padding != "same" and (
            isinstance(padding, bool)
            or not isinstance(padding, int)
            or any(padding != (k - 1) // 2 for k in kernel_sizes)
        ):
            raise ValueError(
                f"padding must be 'same' or (k - 1) // 2 for every kernel in {kernel_sizes}; "
                f"got {padding!r}. PixelShuffle multiplies whatever size a convolution loses "
                f"by scale."
            )

        self._hparams = {
            "scale": scale,
            "in_out_channels": in_out_channels,
            "filters": filters,
            "kernel_sizes": kernel_sizes,
            "padding": padding,
        }

        widths = (in_out_channels, *filters)
        layers: list[torch.nn.Module] = []
        for c_in, c_out, k in zip(widths[:-1], filters, kernel_sizes[:-1], strict=True):
            layers += [torch.nn.Conv2d(c_in, c_out, k, padding=padding), torch.nn.Tanh()]
        layers += [
            torch.nn.Conv2d(
                widths[-1], in_out_channels * scale**2, kernel_sizes[-1], padding=padding
            ),
            torch.nn.PixelShuffle(scale),
        ]
        self.net = torch.nn.Sequential(*layers)

    @property
    def variant_tag(self) -> str:
        """Kernel sizes, e.g. ``'5-3-3'`` -- the paper's own shorthand for the layout."""
        return "-".join(str(k) for k in self._hparams["kernel_sizes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Args:
            x: Native-resolution input, shape ``(B, in_out_channels, H, W)``.

        Returns:
            Output of shape ``(B, in_out_channels, H*scale, W*scale)``.
        """
        return self.net(x)
