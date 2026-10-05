"""DRCN -- the deeply-recursive convolutional network of Kim, Lee & Lee (2016).

Reference: Deeply-Recursive Convolutional Network for Image Super-Resolution
(https://arxiv.org/abs/1511.04491).
"""

from typing import Any, ClassVar, Literal

import torch

from sisr.models.base import SRModel, SRModelOutput


class DRCN(SRModel):
    """Embedding, a shared recursive block, a shared reconstruction net, a learned ensemble.

    This is the wiring of the authors' released network, which the pretrained
    weights require. The embedding net (two 3x3 convs with ReLUs) produces
    ``H0``. A **shared** 3x3 conv, ``recon1``, maps it to ``z_0`` (linear); then
    ``recursions`` times, ``z_d = recon1(relu(infer(z_{d-1})))`` with ``infer`` a
    second shared 3x3 conv, so the parameter count is independent of
    ``recursions``. A final 3x3 conv, ``recon2``, reads every ``z_d`` and gives
    a residual ``r_d``: ``recursions + 1`` predictions. The output adds the
    input skip once, after the learned ensemble: ``x + sum_d w_d * r_d``.

    That equals the paper's ``sum_d w_d * (x + r_d)`` only when the weights sum
    to 1, as they do at initialisation (``1 / (recursions + 1)`` each); the
    released weights sum to about 1.65, so the skip placement matters for
    parity. **The ensemble weights are raw and un-normalised** (no softmax).

    ``forward`` returns an :class:`~sisr.models.base.SRModelOutput`: ``primary``
    is the output above, ``extras["recursions"]`` stacks every per-recursion
    prediction ``x + r_d`` as ``(B, recursions + 1, C, H, W)`` for recursive
    supervision (its MSE to the target equals the residual MSE the authors
    train on). Metrics, predict and export read only ``primary``.

    Reference:
    - Deeply-Recursive Convolutional Network for Image Super-Resolution:
      https://arxiv.org/abs/1511.04491

    Args:
        in_out_channels: Input/output channel count (1 for the paper's Y channel).
        features: Width of every hidden conv (256 in the paper).
        recursions: Applications of the shared inference conv (16 in the paper).
        kernel_size: Odd kernel size of every conv (3 in the paper); padding
            keeps the resolution.

    Raises:
        ValueError: If any size is not a positive int, or ``kernel_size`` is even.
    """

    #: Resolution-preserving: LR arrives already bicubic-upsampled to HR size.
    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]] = "pre_upsampled"

    def __init__(
        self,
        in_out_channels: int = 1,
        features: int = 256,
        recursions: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        for name, value in (
            ("in_out_channels", in_out_channels),
            ("features", features),
            ("recursions", recursions),
            ("kernel_size", kernel_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int; got {value!r}.")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd so 'same' padding keeps the resolution; "
                f"got {kernel_size}."
            )

        self._hparams = {
            "in_out_channels": in_out_channels,
            "features": features,
            "recursions": recursions,
            "kernel_size": kernel_size,
        }
        self.recursions = recursions

        def conv(c_in: int, c_out: int) -> torch.nn.Conv2d:
            return torch.nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size // 2)

        self.embed = torch.nn.Sequential(
            conv(in_out_channels, features),
            torch.nn.ReLU(inplace=True),
            conv(features, features),
            torch.nn.ReLU(inplace=True),
        )
        self.infer = conv(features, features)
        self.recon1 = conv(features, features)  # shared: also sits inside the recursion
        self.recon2 = conv(features, in_out_channels)
        self.ensemble = torch.nn.Parameter(torch.full((recursions + 1,), 1.0 / (recursions + 1)))

    @property
    def variant_tag(self) -> str:
        """The recursion depth, e.g. ``'r16'``."""
        return f"r{self._hparams['recursions']}"

    def reset_parameters(self, **kwargs: Any) -> None:
        """The paper's init: He for ordinary convs, identity for the shared ones.

        Non-recursive convs use He et al.'s normal init. The two shared
        convs, ``infer`` and ``recon1``, are zero except for the
        self-connection (the centre tap from each channel to the same channel),
        as in the authors' released network. All biases are zero. The ensemble
        weights are reset to ``1 / (recursions + 1)``.

        Args:
            **kwargs: Unused; absorbed so callers can invoke this
                polymorphically across architectures.
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                if module is self.infer or module is self.recon1:
                    torch.nn.init.zeros_(module.weight)
                    centre = module.kernel_size[0] // 2
                    with torch.no_grad():
                        for c in range(module.out_channels):
                            module.weight[c, c, centre, centre] = 1.0
                else:
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                assert module.bias is not None
                torch.nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.ensemble.fill_(1.0 / (self.recursions + 1))

    def forward(self, x: torch.Tensor) -> SRModelOutput:
        """Run the recursions and the learned ensemble.

        Args:
            x: Pre-upsampled input, ``(B, in_out_channels, H, W)``.

        Returns:
            An :class:`~sisr.models.base.SRModelOutput` whose ``primary`` is
            ``(B, in_out_channels, H, W)`` and whose ``extras["recursions"]``
            is ``(B, recursions + 1, in_out_channels, H, W)``.
        """
        z = self.recon1(self.embed(x))
        residuals = [self.recon2(z)]
        for _ in range(self.recursions):
            z = self.recon1(torch.relu(self.infer(z)))
            residuals.append(self.recon2(z))
        stacked = torch.stack(residuals, dim=1)
        primary = x + (stacked * self.ensemble.view(1, -1, 1, 1, 1)).sum(dim=1)
        return SRModelOutput(primary=primary, extras={"recursions": x.unsqueeze(1) + stacked})
