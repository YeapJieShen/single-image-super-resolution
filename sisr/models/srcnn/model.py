import torch


class SRCNN(torch.nn.Module):
    """
    SRCNN (Super-Resolution Convolutional Neural Network) model for single image super-resolution.
    The architecture consists of three main parts: feature extraction, non-linear mapping, and reconstruction.

    Reference:
    - Image Super-Resolution Using Deep Convolutional Networks: https://arxiv.org/pdf/1501.00092

    Args:
        num_channels (int): The number of channels in the input and output images (e.g., 3 for RGB, 1 for Y channel).
        num_filters (tuple[int, ...]): A tuple containing the number of filters for each convolutional layer
            (e.g., (64, 32, 1) for the original SRCNN architecture).
        kernel_sizes (tuple[int, ...]): A tuple containing the kernel sizes for each convolutional layer
            (e.g., (9, 1, 5) for the original SRCNN architecture).
        padding (str | int): The padding type or size for the convolutional layers.
            Can be 'valid', 'same', or an integer specifying the number of pixels to pad. Default is 'valid'.
        custom_init (bool): Whether to use custom weight initialization for the convolutional layers. Default
            is False, which uses the default initialization method in PyTorch.
        init_mean (float): The mean of the normal distribution for custom weight initialization. Default is 0.0. Only used if custom_init is True.
        init_std (float): The standard deviation of the normal distribution for custom weight initialization. Default is 0.01. Only used if custom_init is True.
    """

    def __init__(self, num_channels: int, num_filters: tuple[int, ...], kernel_sizes: tuple[int, ...], padding: str | int = 'valid', custom_init: bool = False, init_mean: float = 0.0, init_std: float = 0.01):
        super().__init__()

        self._check_architecture(num_filters, kernel_sizes)

        self._hparams = {
            'num_channels': num_channels,
            'num_filters': num_filters,
            'kernel_sizes': kernel_sizes,
            'padding': padding,
            'custom_init': custom_init,
            'init_mean': init_mean,
            'init_std': init_std,
        }

        self.feat = torch.nn.Sequential(
            torch.nn.Conv2d(
                num_channels, num_filters[0], kernel_size=kernel_sizes[0], padding=padding),
            torch.nn.ReLU(inplace=True)
        )

        self.mapping = torch.nn.Sequential(
            *[
                layer
                # Skip 1st and last layer, defined separately
                for i in range(len(num_filters) - 1)
                for layer in (
                    torch.nn.Conv2d(
                        num_filters[i], num_filters[i + 1], kernel_size=kernel_sizes[i + 1], padding=padding),
                    torch.nn.ReLU(inplace=True)
                )
            ]
        )

        self.recon = torch.nn.Conv2d(
            num_filters[-1], num_channels, kernel_size=kernel_sizes[-1], padding=padding)

        if custom_init:
            self.reset_parameters(mean=init_mean, std=init_std)

    def _check_architecture(self, num_filters: tuple[int, ...], kernel_sizes: tuple[int, ...]):
        """
        Validates the architecture parameters for the SRCNN model.

        Args:
            num_filters (tuple[int, ...]): A tuple containing the number of filters for each convolutional layer
            kernel_sizes (tuple[int, ...]): A tuple containing the kernel sizes for each convolutional layer
            padding (str | int): The padding type or size for the convolutional layers

        Raises:
            ValueError: If num_filters or kernel_sizes are not tuples, if they have different lengths, or if any of their elements are not positive integers.
        """
        for i, name in zip([num_filters, kernel_sizes], ["num_filters", "kernel_sizes"]):
            if not isinstance(i, tuple):
                raise ValueError(f"{name} must be tuples. Got {type(i)}.")
            elif len(i) == 0:
                raise ValueError(f"{name} cannot be empty. Got {i}.")
            elif len(i) < 3:
                if len(i) < 2 and name == "num_filters":
                    raise ValueError(
                        f"{name} must have at least 2 elements for the feature extraction and reconstruction layers. Got {i}.")
                elif name == "kernel_sizes":
                    raise ValueError(
                        f"{name} must have at least 3 elements for the feature extraction, mapping, and reconstruction layers. Got {i}.")
            if any(f < 1 for f in i):
                raise ValueError(
                    f"All elements in {name} must be positive integers. Got {i}.")

        if len(num_filters) + 1 != len(kernel_sizes):
            raise ValueError(
                f"num_filters must have exactly one less element than kernel_sizes. Got num_filters={num_filters} and kernel_sizes={kernel_sizes}.")

    @property
    def hparams(self) -> dict:
        return self._hparams

    def reset_parameters(self, mean: float = 0.0, std: float = 0.01):
        """
        Initializes the weights of the convolutional layers using a normal distribution and sets the biases to zero.
        Following the initialization method described in the original SRCNN paper (https://arxiv.org/pdf/1501.00092).

        Args:
            mean (float): The mean of the normal distribution for weight initialization. Default is 0
            std (float): The standard deviation of the normal distribution for weight initialization. Default is 0.01
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=mean, std=std)
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor, clamp_output: bool = False, clamp_minmax: tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
        """
        Forward pass of the SRCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width)
            clamp_output (bool): Whether to clamp the output values to a specified range. Default is False.
            clamp_minmax (tuple[float, float]): The minimum and maximum values for clamping the output. Default is (0.0, 1.0).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_channels, height, width)
        """
        x = self.feat(x)
        x = self.mapping(x)
        x = self.recon(x)

        if clamp_output:
            x = torch.clamp(x, min=clamp_minmax[0], max=clamp_minmax[1])

        return x
