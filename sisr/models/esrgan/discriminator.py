"""ESRGAN's discriminator -- a VGG-style classifier on 128x128 HR patches.

The paper (Wang et al. 2018) does not draw its discriminator; this is the VGG-128 network of
the authors' released training code (BasicSR's ``basicsr/archs/discriminator_arch.py``,
``VGGStyleDiscriminator``), layer for layer at 128. Away from 128 this differs from it: it
keeps five stages and widens the dense head, where BasicSR asserts 128 or 256 and adds a stage
at 256. The layout: a 3x3 stem, then five stages of a 3x3 stride-1 conv and a 4x4 stride-2
conv (channels 64, 128, 256, 512, 512) with BatchNorm, and two dense layers of 100 and 1.
Four-by-four stride-2 convs, not the 3x3 stride-2 of SRGAN's discriminator, are the visible
difference.
"""

import torch

from sisr.models.srgan.base import AdversarialDiscriminator

_STAGES = (64, 128, 256, 512, 512)
_DOWNSAMPLES = len(_STAGES)


class VGGStyleDiscriminator(AdversarialDiscriminator):
    """VGG-style critic emitting **logits**; pairs with a relativistic-average loss.

    Args:
        in_channels: Input channels (3, RGB).
        hr_input_size: Square patch size; a positive multiple of 32 (five stride-2 convs). The
            dense head fixes it, so it is declared and checked against the training crop at
            setup. 128 in the paper.
        base_channels: Stem width (64); the stages are ``1, 2, 4, 8, 8`` times it.
        dense_features: Width of the first dense layer (100).
        negative_slope: LeakyReLU slope (0.2).

    Raises:
        ValueError: If ``hr_input_size`` is not a positive multiple of 32.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hr_input_size: int = 128,
        base_channels: int = 64,
        dense_features: int = 100,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        factor = 2**_DOWNSAMPLES
        if hr_input_size < factor or hr_input_size % factor:
            raise ValueError(
                f"hr_input_size must be a positive multiple of {factor} (the discriminator has "
                f"{_DOWNSAMPLES} stride-2 convolutions); got {hr_input_size}."
            )
        self._hparams = {
            "in_channels": in_channels,
            "hr_input_size": hr_input_size,
            "base_channels": base_channels,
            "dense_features": dense_features,
            "negative_slope": negative_slope,
        }
        act = lambda: torch.nn.LeakyReLU(negative_slope, inplace=True)  # noqa: E731
        layers: list[torch.nn.Module] = [
            torch.nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            act(),
        ]
        channels = base_channels
        for i, mult in enumerate((1, 2, 4, 8, 8)):
            width = base_channels * mult
            if i > 0:  # the first stage's 3x3 keeps the stem's width (and its bias)
                layers += [
                    torch.nn.Conv2d(channels, width, 3, 1, 1, bias=False),
                    torch.nn.BatchNorm2d(width),
                    act(),
                ]
            layers += [
                torch.nn.Conv2d(width, width, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(width),
                act(),
            ]
            channels = width
        self.features = torch.nn.Sequential(*layers)
        spatial = hr_input_size // factor
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(channels * spatial * spatial, dense_features),
            act(),
            torch.nn.Linear(dense_features, 1),
        )

    @property
    def in_channels(self) -> int:
        """Input channel count."""
        return self._hparams["in_channels"]

    @property
    def input_size(self) -> int:
        """Fixed spatial size (height and width) this discriminator accepts."""
        return self._hparams["hr_input_size"]

    @property
    def variant_tag(self) -> str:
        """The HR input size it was built for."""
        return str(self._hparams["hr_input_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of HR-sized images; returns ``(B, 1)`` logits (no sigmoid)."""
        return self.classifier(self.features(x).flatten(1))
