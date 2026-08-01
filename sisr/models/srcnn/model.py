"""SRCNN — the 3-layer CNN from Dong et al. (2014).

Reference: Image Super-Resolution Using Deep Convolutional Networks
(https://arxiv.org/pdf/1501.00092).
"""

import torch

from sisr.models.base import SRModel


class SRCNN(SRModel):
    """SRCNN: feature extraction, non-linear mapping, and reconstruction, in three conv stacks.

    Reference:
    - Image Super-Resolution Using Deep Convolutional Networks: https://arxiv.org/pdf/1501.00092

    Args:
        num_channels: Input/output channel count (e.g. 3 for RGB, 1 for Y).
        num_filters: Filter count per conv layer (e.g. ``(64, 32, 1)`` for
            the original architecture).
        kernel_sizes: Kernel size per conv layer (e.g. ``(9, 1, 5)`` for the
            original architecture).
        padding: ``'valid'``, ``'same'``, or an explicit pixel count.
            Defaults to ``'valid'``.
    """

    def __init__(
        self,
        num_channels: int,
        num_filters: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        padding: str | int = "valid",
    ):
        super().__init__()

        self._check_architecture(num_filters, kernel_sizes)

        self._hparams = {
            "num_channels": num_channels,
            "num_filters": num_filters,
            "kernel_sizes": kernel_sizes,
            "padding": padding,
        }

        self.feat = torch.nn.Sequential(
            torch.nn.Conv2d(
                num_channels, num_filters[0], kernel_size=kernel_sizes[0], padding=padding
            ),
            torch.nn.ReLU(inplace=True),
        )

        self.mapping = torch.nn.Sequential(
            *[
                layer
                for i in range(len(num_filters) - 1)
                for layer in (
                    torch.nn.Conv2d(
                        num_filters[i],
                        num_filters[i + 1],
                        kernel_size=kernel_sizes[i + 1],
                        padding=padding,
                    ),
                    torch.nn.ReLU(inplace=True),
                )
            ]
        )

        self.recon = torch.nn.Conv2d(
            num_filters[-1], num_channels, kernel_size=kernel_sizes[-1], padding=padding
        )

    def _check_architecture(self, num_filters: tuple[int, ...], kernel_sizes: tuple[int, ...]):
        """Validates num_filters/kernel_sizes are same-length positive-int tuples."""
        for i, name in zip(
            [num_filters, kernel_sizes], ["num_filters", "kernel_sizes"], strict=False
        ):
            if not isinstance(i, tuple):
                raise ValueError(f"{name} must be tuples. Got {type(i)}.")
            elif len(i) == 0:
                raise ValueError(f"{name} cannot be empty. Got {i}.")
            elif len(i) < 3:
                if len(i) < 2 and name == "num_filters":
                    raise ValueError(
                        f"{name} must have at least 2 elements for the feature extraction "
                        f"and reconstruction layers. Got {i}."
                    )
                elif name == "kernel_sizes":
                    raise ValueError(
                        f"{name} must have at least 3 elements for the feature extraction, "
                        f"mapping, and reconstruction layers. Got {i}."
                    )
            if any(f < 1 for f in i):
                raise ValueError(f"All elements in {name} must be positive integers. Got {i}.")

        if len(num_filters) + 1 != len(kernel_sizes):
            raise ValueError(
                f"num_filters must have exactly one less element than kernel_sizes. "
                f"Got num_filters={num_filters} and kernel_sizes={kernel_sizes}."
            )

    def reset_parameters(self, mean: float = 0.0, std: float = 0.01):
        """Gaussian weight init + zero biases, per the SRCNN paper (https://arxiv.org/pdf/1501.00092).

        Args:
            mean: Mean of the weight-init normal distribution.
            std: Standard deviation of the weight-init normal distribution.
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=mean, std=std)
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        clamp_output: bool = False,
        clamp_minmax: tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """Forward pass: feature extraction -> non-linear mapping -> reconstruction.

        clamp_output is a direct-call convenience for offline inference: the
        SRLightning training and validation paths always call model(x) without
        it, so clamping only takes effect on direct model(x, clamp_output=True)
        calls used to clip the raw output into a displayable range.

        Args:
            x: Input tensor, shape ``(B, num_channels, H, W)``.
            clamp_output: Whether to clamp the output to clamp_minmax. Only
                honoured on direct ``model(x, clamp_output=True)`` calls; the
                SRLightning pipeline never sets it.
            clamp_minmax: Min/max values for clamping the output.

        Returns:
            Output tensor, same shape as *x*.
        """
        x = self.feat(x)
        x = self.mapping(x)
        x = self.recon(x)

        if clamp_output:
            x = torch.clamp(x, min=clamp_minmax[0], max=clamp_minmax[1])

        return x
