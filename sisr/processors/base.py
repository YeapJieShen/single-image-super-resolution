"""Abstract base class for per-batch colorspace adapters."""

import abc

import torch


class SRProcessor(abc.ABC):
    """Adapts dataset-format tensors (RGB float in [0, 1]) to/from model IO.

    Partitioned by colorspace, not by model — the same processor instance
    is shared across all models that train in that colorspace. Replaces
    the per-batch colorspace logic that previously lived as standalone
    functions in :mod:`sisr.colorspace`.
    """

    @abc.abstractmethod
    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        """Convert dataset LR (RGB, ``[B, 3, H, W]``) into the model's input tensor."""

    def extract_target(self, hr_rgb: torch.Tensor) -> torch.Tensor:
        """Convert dataset HR (RGB, ``[B, 3, H', W']``) into the model's *output* space.

        The training loss is computed between ``model(extract(lr))`` and
        ``extract_target(hr)``, so an implementation must be the exact
        inverse of :meth:`reconstruct`.

        Defaults to :meth:`extract`, which is correct for every processor
        whose input and output spaces coincide — i.e. every pure colorspace
        adapter, where the same conversion applies to LR and HR alike.
        Override only when the model consumes and emits *different* ranges;
        :class:`~sisr.processors.rgb.RGBSignedOutputProcessor` is the one
        such case, and exists because the SRGAN paper specifies exactly that
        asymmetry.

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

        A fact only the processor knows (not a mirror of the model's own
        config), used to validate a model/processor pairing at construction
        time (see ``SRTrainingConfig.validate_against``).
        """
