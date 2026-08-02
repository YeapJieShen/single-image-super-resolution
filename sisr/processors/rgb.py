"""RGB processors — model trains and emits RGB, in one of two output ranges."""

import torch

from .base import SRProcessor


class RGBProcessor(SRProcessor):
    """No-op processor: model trains and emits RGB directly."""

    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        return lr_rgb

    def reconstruct(self, sr_model_out: torch.Tensor, lr_rgb: torch.Tensor) -> torch.Tensor:
        return sr_model_out

    @property
    def model_channels(self) -> int:
        return 3


class RGBSignedOutputProcessor(SRProcessor):
    """RGB in and out, with the model's *output* space in ``[-1, 1]``.

    Ledig et al. §3.2: "We scaled the range of the LR input images to [0, 1]
    and for the HR images to [-1, 1]. The MSE loss was thus calculated on
    images of intensity range [-1, 1]." The asymmetry is the point — the
    model consumes ``[0, 1]`` and emits ``[-1, 1]`` — which is why the target
    mapping lives in :meth:`extract_target` rather than :meth:`extract`.

    Practically this centres the regression target on zero, which matches
    what a freshly initialised conv stack emits; against a ``[0, 1]`` target
    the tail conv has to learn a +0.5 mean shift first. The 4x it also
    applies to the MSE is *not* a second effect — Adam normalises by the
    second moment, so a global loss scaling cancels out of the update. Only
    the logged ``loss/train`` value differs (by exactly 4x).

    Pair with an unscaled :class:`RGBProcessor` to reproduce runs recorded
    before this processor existed.
    """

    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        return lr_rgb

    def extract_target(self, hr_rgb: torch.Tensor) -> torch.Tensor:
        return hr_rgb * 2.0 - 1.0

    def reconstruct(self, sr_model_out: torch.Tensor, lr_rgb: torch.Tensor) -> torch.Tensor:
        return (sr_model_out + 1.0) / 2.0

    @property
    def model_channels(self) -> int:
        return 3
