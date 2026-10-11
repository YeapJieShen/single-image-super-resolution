"""MDSR -- the multi-scale deep super-resolution network of Lim et al. (2017).

Reference: Enhanced Deep Residual Networks for Single Image Super-Resolution
(https://arxiv.org/abs/1707.02921), Section 3.2 and Figure 5.
"""

import math
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import torch

from sisr.models.base import SRModel
from sisr.models.edsr.model import DIV2K_RGB_MEAN, ResBlock


class MDSR(SRModel):
    """One shared residual trunk with scale-specific pre-processing and upsampling.

    ``head`` conv -> a **per-scale** pair of residual blocks with 5x5 convs (``pre_process``)
    -> a **shared** trunk of ``num_blocks`` residual blocks and a closing conv, added back onto
    the pre-processed features -> a **per-scale** upsampler (``conv`` + ``PixelShuffle``: one
    stage for x2 and x3, two for x4) -> a shared ``tail`` conv. No residual scaling and no
    BatchNorm. Only the modules of the requested scale run, so a batch has one scale
    (the training loop draws one per batch).

    ``multi_scale`` is true: :class:`~sisr.training.SRLightning` calls ``forward(x, scale=s)``
    with the batch's factor. As in :class:`~sisr.models.edsr.EDSR` the model takes and returns
    [0, 1] and applies the 0-255 range and dataset mean shift internally, as fixed buffers.

    Presets from the paper: **baseline** ``num_blocks=16`` (3.2M parameters) and **MDSR**
    ``num_blocks=80`` (8.0M), both with 64 features. MDSR+ is this model with
    ``eval_config.self_ensemble: true``.

    Reference:
    - Enhanced Deep Residual Networks for Single Image Super-Resolution:
      https://arxiv.org/abs/1707.02921

    Args:
        scale: The factors served, in the order the released checkpoint indexes them
            (``[2, 3, 4]``); each is 3 or a power of two. Recorded as the ``scale`` hparam, so a
            ``training_config.scale`` list is validated against it.
        in_out_channels: Input/output channel count (3, RGB).
        features: Trunk width (64).
        num_blocks: Shared residual block count (16 or 80).
        rgb_mean: Per-channel mean in [0, 1].
        rgb_range: The range the weights were trained at (255).

    Raises:
        ValueError: If a size is not a positive int, ``scale`` is empty, repeats a factor or holds
            one that is neither 3 nor a power of two, or ``rgb_mean`` has the wrong length.
    """

    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]] = "native_lr"
    multi_scale: ClassVar[bool] = True
    mean: torch.Tensor

    def __init__(
        self,
        scale: Sequence[int] = (2, 3, 4),
        in_out_channels: int = 3,
        features: int = 64,
        num_blocks: int = 16,
        rgb_mean: tuple[float, ...] = DIV2K_RGB_MEAN,
        rgb_range: float = 255.0,
    ):
        super().__init__()
        scales = list(scale)
        for name, value in (
            ("in_out_channels", in_out_channels),
            ("features", features),
            ("num_blocks", num_blocks),
            *[("scale entry", s) for s in scales],
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int; got {value!r}.")
        if not scales or len(set(scales)) != len(scales):
            raise ValueError(f"scale must list distinct factors; got {scales}.")
        for s in scales:
            if s != 3 and (s < 2 or s & (s - 1)):
                raise ValueError(f"each scale must be 3 or a power of two >= 2; got {s}.")
        if len(rgb_mean) != in_out_channels:
            raise ValueError(f"rgb_mean needs {in_out_channels} entries; got {tuple(rgb_mean)}.")
        if rgb_range <= 0:
            raise ValueError(f"rgb_range must be positive; got {rgb_range}.")

        self._hparams = {
            "scale": scales,
            "in_out_channels": in_out_channels,
            "features": features,
            "num_blocks": num_blocks,
            "rgb_mean": list(rgb_mean),
            "rgb_range": rgb_range,
        }
        self.scales = scales
        self.rgb_range = rgb_range
        self.register_buffer(
            "mean", torch.tensor(rgb_mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False
        )
        self.head = torch.nn.Conv2d(in_out_channels, features, 3, padding=1)
        self.pre_process = torch.nn.ModuleList(
            torch.nn.Sequential(ResBlock(features, 1.0, 5), ResBlock(features, 1.0, 5))
            for _ in scales
        )
        self.blocks = torch.nn.Sequential(*[ResBlock(features, 1.0) for _ in range(num_blocks)])
        self.body_conv = torch.nn.Conv2d(features, features, 3, padding=1)
        self._factors = [[3] if s == 3 else [2] * int(math.log2(s)) for s in scales]
        self.up = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.Conv2d(features, features * f * f, 3, padding=1) for f in factors
            )
            for factors in self._factors
        )
        self.tail = torch.nn.Conv2d(features, in_out_channels, 3, padding=1)

    @property
    def variant_tag(self) -> str:
        """Blocks and width, e.g. ``'16B64F'`` or ``'80B64F'``."""
        return f"{self._hparams['num_blocks']}B{self._hparams['features']}F"

    def reset_parameters(self, **kwargs: Any) -> None:
        """No-op: the paper states no initialisation, so PyTorch's default stands.

        Args:
            **kwargs: Unused; absorbed so callers can invoke this polymorphically.
        """

    def forward(self, x: torch.Tensor, scale: int | None = None) -> torch.Tensor:
        """Upscale ``x`` by ``scale``.

        Args:
            x: LR batch in [0, 1], ``(B, in_out_channels, H, W)``.
            scale: Which factor to run; ``None`` is the first one listed (for direct calls).

        Returns:
            ``(B, in_out_channels, H * scale, W * scale)`` in [0, 1] (unclamped).

        Raises:
            ValueError: If ``scale`` is not one the model was built for.
        """
        if scale is None:
            scale = self.scales[0]
        if scale not in self.scales:
            raise ValueError(f"scale {scale} is not one of this model's scales {self.scales}.")
        i = self.scales.index(scale)
        h = self.pre_process[i](self.head((x - self.mean) * self.rgb_range))
        h = h + self.body_conv(self.blocks(h))
        convs: Any = self.up[i]
        for conv, factor in zip(convs, self._factors[i], strict=True):
            h = torch.nn.functional.pixel_shuffle(conv(h), factor)
        return self.tail(h) / self.rgb_range + self.mean
