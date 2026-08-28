"""SRGAN's discriminator — the classifier from Ledig et al. (2017), Figure 4.

Reference: Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network (https://arxiv.org/pdf/1609.04802).
"""

import torch

#: (out_channels_multiplier, stride) for the 7 blocks after the stem, per Fig. 4.
_BLOCKS: tuple[tuple[int, int], ...] = ((1, 2), (2, 1), (2, 2), (4, 1), (4, 2), (8, 1), (8, 2))

#: Number of stride-2 convolutions above; the spatial reduction factor is 2**this.
_DOWNSAMPLES = sum(1 for _, stride in _BLOCKS if stride == 2)


class SRDiscriminator(torch.nn.Module):
    """SRGAN discriminator: conv stem, 7 strided conv-BN-LeakyReLU blocks, dense head.

    Deliberately **not** an :class:`~sisr.models.base.SRModel`: it is a
    classifier, not an SR mapping, so ``input_contract`` and
    ``reset_parameters`` would be contracts it cannot honour. It carries
    ``hparams`` only so provenance metadata can describe it the same way.

    **Emits logits, not probabilities.** Ledig's Figure 4 ends in a sigmoid;
    pairing an unactivated output with ``BCEWithLogitsLoss`` computes the same
    objective in a numerically stable way. Do not add the sigmoid back — with
    the loss this project pairs it with, that double-activates.

    The dense head fixes the accepted input size, which is why
    ``hr_input_size`` is declared rather than inferred: a later task validates
    it against the real training crop, so a mismatch there is a readable
    error instead of a raw ``Linear`` shape failure mid-run.

    Args:
        in_channels: Input channel count. Defaults to ``3`` (RGB).
        hr_input_size: Spatial size of the square HR patch this discriminator
            accepts. Must be divisible by 16 (four stride-2 convolutions).
            Defaults to ``96`` — the paper's crop.
        base_channels: Channel count of the stem; later blocks are multiples of
            it, per Figure 4. Defaults to ``64``.
        dense_features: Width of the first dense layer. Defaults to ``1024``,
            the paper's value.
        negative_slope: LeakyReLU slope. Defaults to ``0.2``, the paper's value.

    Raises:
        ValueError: If ``hr_input_size`` is not a positive multiple of 16.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hr_input_size: int = 96,
        base_channels: int = 64,
        dense_features: int = 1024,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        stride_factor = 2**_DOWNSAMPLES
        if hr_input_size < stride_factor or hr_input_size % stride_factor:
            raise ValueError(
                f"hr_input_size must be a positive multiple of {stride_factor} "
                f"(divisible by {stride_factor}; the discriminator has "
                f"{_DOWNSAMPLES} stride-2 convolutions); got {hr_input_size}."
            )

        self._hparams = {
            "in_channels": in_channels,
            "hr_input_size": hr_input_size,
            "base_channels": base_channels,
            "dense_features": dense_features,
            "negative_slope": negative_slope,
        }

        layers: list[torch.nn.Module] = [
            torch.nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope, inplace=True),
        ]
        channels = base_channels
        for multiplier, stride in _BLOCKS:
            out_channels = base_channels * multiplier
            layers += [
                torch.nn.Conv2d(channels, out_channels, kernel_size=3, stride=stride, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(negative_slope, inplace=True),
            ]
            channels = out_channels
        self.features = torch.nn.Sequential(*layers)

        spatial = hr_input_size // stride_factor
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(channels * spatial * spatial, dense_features),
            torch.nn.LeakyReLU(negative_slope, inplace=True),
            torch.nn.Linear(dense_features, 1),
        )

    @property
    def hparams(self) -> dict:
        """Architecture hyperparameters, for provenance metadata."""
        return self._hparams

    @property
    def variant_tag(self) -> str:
        """The HR input size it was built for -- the one knob that must match the data.

        Not an ``SRModel``, so this is duck-typed rather than inherited. The
        artifact naming reads it the same way either way.
        """
        return str(self._hparams["hr_input_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of HR-sized images.

        Args:
            x: ``(B, in_channels, hr_input_size, hr_input_size)`` in the
                generator's **model output space** (e.g. ``[-1, 1]`` under
                ``RGBSignedOutputProcessor``), not display range.

        Returns:
            ``(B, 1)`` **logits** — higher means "more real". Feed these to
            ``BCEWithLogitsLoss``; do not apply a sigmoid first.
        """
        return self.classifier(self.features(x).flatten(1))
