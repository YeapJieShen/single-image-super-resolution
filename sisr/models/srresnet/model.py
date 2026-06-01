"""SRResNet — the residual generator from Ledig et al. (2017).

Reference: Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network (https://arxiv.org/pdf/1609.04802).
"""
import torch
import math


class SRResidualBlock(torch.nn.Module):
    """A single residual block used in SRResNet.
    Each block applies two convolutional layers with batch normalization,
    and adds the input (identity) as a skip connection.

    Args:
        channels (int): Number of input and output channels.
        kernel_size (int): Kernel size for both convolutional layers.
        padding (str | int): Padding for the convolutional layers. Default is 'same'.
    """

    def __init__(self, channels: int, kernel_size: int, padding: str | int = 'same'):
        super().__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(channels),
            torch.nn.PReLU()
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        identity = x
        x = self.block1(x)
        x = self.block2(x)
        return identity + x


class SRUpsampleBlock(torch.nn.Module):
    """An upsampling block used in SRResNet that increases spatial resolution using sub-pixel convolution.

    Args:
        channels (int): Number of input channels.
        scale (int): Upscaling factor. Default is 2.
        kernel_size (int): Kernel size for the convolutional layer. Default is 3.
    """

    def __init__(self, channels: int, scale: int = 2, kernel_size: int = 3):
        super().__init__()

        self.upsample = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels * (scale ** 2), kernel_size=kernel_size, padding='same'),
            torch.nn.PixelShuffle(scale),
            torch.nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the upsample block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with height and width scaled by the upscaling factor.
        """
        return self.upsample(x)


class SRResNet(torch.nn.Module):
    """SRResNet (Super-Resolution Residual Network) model for single image super-resolution.
    The architecture consists of an initial feature extraction block, a series of residual blocks
    with a skip connection, sub-pixel upsample blocks, and a final reconstruction layer.

    Reference:
    - Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network: https://arxiv.org/pdf/1609.04802

    Args:
        scale (int): The upscaling factor. Must be a power of 2.
        in_out_channels (int): Number of channels in the input and output images (e.g., 3 for RGB). Default is 3.
        hidden_channel (int): Number of feature channels used in the residual and upsample blocks. Default is 64.
        kernel_sizes (tuple[int, ...]): Kernel sizes for the head, residual, and tail convolutional layers. Default is (9, 3, 9).
        num_residual_blocks (int): Number of residual blocks in the network. Default is 16.
        padding (str | int): Padding for the convolutional layers. Default is 'same'.
    """

    def __init__(self, scale: int, in_out_channels: int = 3, hidden_channel: int = 64, kernel_sizes: tuple[int, ...] = (9, 3, 9), num_residual_blocks: int = 16, padding: str | int = 'same'):
        super().__init__()

        self._check_scale(scale)

        self._hparams = {
            'scale': scale,
            'in_out_channels': in_out_channels,
            'hidden_channel': hidden_channel,
            'kernel_sizes': kernel_sizes,
            'num_residual_blocks': num_residual_blocks,
            'padding': padding,
        }

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(in_out_channels, hidden_channel, kernel_size=kernel_sizes[0], padding=padding),
            torch.nn.PReLU()
        )

        self.residual_blocks = torch.nn.Sequential(
            *[SRResidualBlock(channels=hidden_channel, kernel_size=kernel_sizes[1], padding=padding) for _ in range(num_residual_blocks)]
        )

        self.post_residual = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=kernel_sizes[1], padding=padding),
            torch.nn.BatchNorm2d(hidden_channel)
        )

        self.upsample = torch.nn.Sequential(
            *[SRUpsampleBlock(channels=hidden_channel, scale=2) for _ in range(int(math.log2(scale)))]
        )

        self.tail = torch.nn.Conv2d(hidden_channel, in_out_channels, kernel_size=kernel_sizes[2], padding=padding)

    def _check_scale(self, scale: int):
        """Validates that the scale factor is a power of 2.

        Args:
            scale (int): The upscaling factor.

        Raises:
            ValueError: If scale is not a positive integer or not a power of 2.
        """
        if not isinstance(scale, int) or scale < 1:
            raise ValueError(f"scale must be a positive integer. Got {scale}.")
        if (scale & (scale - 1)) != 0:
            raise ValueError(f"scale must be a power of 2. Got {scale}.")

    @property
    def hparams(self) -> dict:
        """Returns the model architecture hyperparameters as a dict.

        Merged into Lightning's HParams by :class:`~sisr.training.SRLightning`
        so the architecture spec appears alongside training params in TensorBoard.
        """
        return self._hparams

    def forward(self, x: torch.Tensor, clamp_output: bool = False, clamp_minmax: tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
        """Forward pass of the SRResNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_out_channels, height, width).
            clamp_output (bool): Whether to clamp the output values to a specified range. Default is False.
            clamp_minmax (tuple[float, float]): The minimum and maximum values for clamping the output. Default is (0.0, 1.0).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_out_channels, height * scale, width * scale).
        """
        x = self.head(x)
        identity = x
        x = self.residual_blocks(x)
        x = identity + self.post_residual(x)
        x = self.upsample(x)
        x = self.tail(x)

        if clamp_output:
            x = torch.clamp(x, min=clamp_minmax[0], max=clamp_minmax[1])

        return x