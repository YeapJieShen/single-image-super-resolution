"""VDSR -- the very deep super-resolution network of Kim, Lee & Lee (2016).

Reference: Accurate Image Super-Resolution Using Very Deep Convolutional Networks
(https://arxiv.org/abs/1511.04587).
"""

from typing import Any, ClassVar, Literal

import torch

from sisr.models.base import SRModel


class VDSR(SRModel):
    """``depth`` 3x3 convolutions with ReLUs between them, and a global residual skip.

    The network predicts the residual between the bicubic-upsampled input and the
    HR image: ``out = x + f(x)``, where ``f`` is ``depth`` same-padded 3x3 convs, a
    ReLU after every one but the last (``features`` wide in between, 64 and 20 in the
    paper). The input is already on the HR grid, so one set of weights serves every
    scale -- the multi-scale model of the paper is just this network trained on
    patches of several scales.

    Reference:
    - Accurate Image Super-Resolution Using Very Deep Convolutional Networks:
      https://arxiv.org/abs/1511.04587

    Args:
        in_out_channels: Input/output channel count (1 for the paper's Y channel).
        depth: Number of convolutions (20 in the paper).
        features: Width of every hidden conv (64 in the paper).
        kernel_size: Odd kernel size of every conv (3 in the paper); padding keeps
            the resolution.

    Raises:
        ValueError: If any size is not a positive int, ``depth`` is below 2, or
            ``kernel_size`` is even.
    """

    #: Resolution-preserving: LR arrives already bicubic-upsampled to HR size, so there
    #: is no scale to hand it (see ``SRModel.multi_scale``).
    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]] = "pre_upsampled"

    def __init__(
        self,
        in_out_channels: int = 1,
        depth: int = 20,
        features: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        for name, value in (
            ("in_out_channels", in_out_channels),
            ("depth", depth),
            ("features", features),
            ("kernel_size", kernel_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int; got {value!r}.")
        if depth < 2:
            raise ValueError(
                f"depth must be at least 2 (an input and an output conv); got {depth}."
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd so 'same' padding keeps the resolution; "
                f"got {kernel_size}."
            )

        self._hparams = {
            "in_out_channels": in_out_channels,
            "depth": depth,
            "features": features,
            "kernel_size": kernel_size,
        }
        pad = kernel_size // 2
        widths = [in_out_channels] + [features] * (depth - 1) + [in_out_channels]
        layers: list[torch.nn.Module] = []
        for i in range(depth):
            layers.append(torch.nn.Conv2d(widths[i], widths[i + 1], kernel_size, padding=pad))
            if i < depth - 1:
                layers.append(torch.nn.ReLU(inplace=True))
        self.body = torch.nn.Sequential(*layers)

    @property
    def variant_tag(self) -> str:
        """The depth, e.g. ``'d20'``."""
        return f"d{self._hparams['depth']}"

    def reset_parameters(self, **kwargs: Any) -> None:
        """He et al.'s init, which is all the paper states about initialisation.

        Every conv's weights are drawn from a zero-mean normal with std
        ``sqrt(2 / (k * k * c_in))`` (He/MSRA, fan-in); biases are zero. The paper says
        only "the method described in He et al." for ReLU networks and does not
        spell out the variant, so fan-in is an assumption.

        Args:
            **kwargs: Unused; absorbed so callers can invoke this
                polymorphically across architectures.
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                torch.nn.init.normal_(module.weight, mean=0.0, std=(2.0 / fan_in) ** 0.5)
                assert module.bias is not None
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the residual and add it to the input.

        Args:
            x: Bicubic-upsampled input, ``(B, in_out_channels, H, W)``.

        Returns:
            ``x + residual``, the same shape.
        """
        return x + self.body(x)
