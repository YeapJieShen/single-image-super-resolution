"""Abstract base class for per-batch colorspace adapters."""

import abc
from typing import Literal

import torch


class SRProcessor(abc.ABC):
    """Adapts dataset-format tensors (RGB float in [0, 1]) to/from model IO.

    Partitioned by colorspace, not by model: one instance serves every model
    training in that colorspace.
    """

    @abc.abstractmethod
    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        """Convert dataset LR (RGB, ``[B, 3, H, W]``) into the model's input tensor."""

    def extract_target(self, hr_rgb: torch.Tensor) -> torch.Tensor:
        """Convert dataset HR (RGB, ``[B, 3, H', W']``) into the model's *output* space.

        The loss is computed between ``model(extract(lr))`` and
        ``extract_target(hr)``, so this must be the exact inverse of
        :meth:`reconstruct`.

        Defaults to :meth:`extract`, correct wherever input and output spaces
        coincide -- every pure colorspace adapter. **Override only when the
        model consumes and emits different ranges**;
        :class:`~sisr.processors.rgb.RGBSignedOutputProcessor` is the one such
        case, and exists because the SRGAN paper specifies that asymmetry.

        Args:
            hr_rgb: HR batch, RGB ``float32`` in ``[0, 1]``.

        Returns:
            The loss target, in the model's output space.
        """
        return self.extract(hr_rgb)

    @abc.abstractmethod
    def reconstruct(self, sr_model_out: torch.Tensor, lr_rgb: torch.Tensor) -> torch.Tensor:
        """Convert the model's output tensor back to SR RGB (``[B, 3, H', W']``)."""

    @property
    @abc.abstractmethod
    def model_channels(self) -> int:
        """Number of channels ``extract`` produces — the model's IO channel count.

        A fact only the processor knows -- not a mirror of the model's config.
        Validates a model/processor pairing at construction
        (``SRTrainingConfig.validate_against``).
        """

    @property
    @abc.abstractmethod
    def output_range(self) -> tuple[float, float]:
        """The model's *output* intensity range, e.g. ``(0.0, 1.0)`` or ``(-1.0, 1.0)``.

        Abstract, never defaulted: a wrong inherited default is the failure
        this exists to prevent. Reconstructing ``[-1, 1]``-trained weights with
        ``[0, 1]`` logic produces silently wrong images and no error. Recorded
        in checkpoint and export metadata so no downstream consumer has to
        guess it.
        """

    @property
    @abc.abstractmethod
    def output_colorspace(self) -> Literal["RGB", "YCbCr", "Y"]:
        """Colorspace of the model's output planes.

        Abstract for the same reason as :attr:`output_range`: nothing tells
        Y/Cb/Cr planes from R/G/B ones by shape, and
        :class:`~sisr.losses.vgg.VGG19FeatureLoss` normalises with RGB ImageNet
        statistics -- it would silently compute a meaningless quantity.
        """
