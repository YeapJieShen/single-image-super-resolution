import torch
import torchvision
import torchmetrics
import lightning
from typing import Optional, List, Dict, Any, Tuple, Literal


class SRLightning(lightning.LightningModule):
    """
    Generic Lightning wrapper for single-image super-resolution models.

    Architecture-agnostic: any ``torch.nn.Module`` that accepts an LR tensor
    of shape ``(B, C, H, W)`` and returns an SR tensor can be plugged in.
    The wrapped model is passed in pre-instantiated (jsonargparse / LightningCLI
    handle the construction via ``class_path`` / ``init_args``).

    Per-layer learning rates (originally needed for SRCNN's recipe) are
    supported via ``layer_optim_params`` and apply to every ``torch.nn.Conv2d``
    found in the wrapped model in module-traversal order.

    Validation steps compute PSNR (configurable per-channel and per-colorspace)
    and SSIM against the high-resolution ground truth (centre-cropped to match
    the SR output when the model uses ``padding='valid'``).

    Args:
        model (torch.nn.Module): An initialised SR model instance.  If the
            model exposes an ``hparams`` mapping (as :class:`~sisr.models.srcnn.SRCNN`
            does), those values are merged into the Lightning hparams so model
            specs appear alongside training params in TensorBoard HParams.
        example_input_shape (Optional[Tuple[int, ...]]): Shape of a single input
            sample *excluding* the batch dimension (e.g. ``(3, 33, 33)`` for a
            33×33 RGB patch).  When provided, sets ``self.example_input_array``
            which Lightning uses to log the model graph to TensorBoard when
            ``TensorBoardLogger(log_graph=True)`` is configured.
        psnr_channels (List[str]): Colorspaces to compute PSNR in.  Supported
            values are ``'RGB'`` and ``'YCbCr'``.  Metrics are logged as
            ``val_psnr_RGB``, ``val_psnr_YCbCr``, etc.  Defaults to ``['RGB']``.
        separate_psnr (bool): When ``True``, also computes PSNR for each
            individual channel within each requested colorspace (e.g. ``'RGB'``
            yields ``val_psnr_R``, ``val_psnr_G``, ``val_psnr_B``, and
            ``val_psnr_RGB``).  Defaults to ``False``.
        lr (float): Base learning rate used when ``layer_optim_params`` is None.
        layer_optim_params (Optional[List[Dict[str, Any]]]): Per-Conv2d-layer
            optimizer overrides (e.g. individual learning rates).  Must have
            the same length as the number of Conv2d layers in ``model``.
            If ``None``, every layer uses the base ``lr``.
        criterion (Optional[torch.nn.Module]): Loss instance.  Defaults to
            :class:`torch.nn.MSELoss` when ``None``.
        optimizer_class (type[torch.optim.Optimizer]): Optimizer class.
            Defaults to :class:`torch.optim.SGD`.
        optimizer_init_args (Optional[Dict[str, Any]]): Extra keyword arguments
            forwarded to the optimizer constructor (e.g. ``momentum``).
        scheduler_class (Optional[type]): LR-scheduler class, or ``None`` to
            disable scheduling.
        scheduler_init_args (Optional[Dict[str, Any]]): Keyword arguments
            forwarded to the scheduler constructor.
        scheduler_interval (Literal['epoch', 'step']): Whether the scheduler
            steps every epoch or every training step.  Defaults to ``'epoch'``.
        model_colorspace (str): The colorspace the model processes.  Controls
            how the LR input is prepared and how the SR output is reconstructed
            back to full RGB for loss and metric computation.

            * ``'RGB'`` *(default)* — model receives and outputs RGB directly.
            * ``'Y'`` — Y channel (from LR YCbCr) is fed to the model; SR output
              is stitched with bicubic Cb/Cr (taken from the LR image) and
              converted back to RGB.  Dataset must serve RGB.
            * ``'YCbCr'`` — model receives and outputs full YCbCr; output is
              converted back to RGB for metrics.  Dataset must serve RGB.

        scheduler_monitor (Optional[str]): Metric to monitor for
            plateau-based schedulers (e.g. ``ReduceLROnPlateau``).
            Defaults to ``None``, which auto-selects ``val_psnr({psnr_channels[0]})``.
    """

    _CS_CHANNEL_NAMES: Dict[str, Tuple[str, ...]] = {
        'RGB':   ('R', 'G', 'B'),
        'YCbCr': ('Y', 'Cb', 'Cr'),
    }

    def __init__(
        self,
        model: torch.nn.Module,
        example_input_shape: Optional[Tuple[int, ...]] = None,
        model_colorspace: str = 'RGB',
        psnr_channels: List[str] = ['RGB'],
        separate_psnr: bool = False,
        lr: float = 1e-4,
        layer_optim_params: Optional[List[Dict[str, Any]]] = None,
        criterion: Optional[torch.nn.Module] = None,
        optimizer_class: type = torch.optim.SGD,
        optimizer_init_args: Optional[Dict[str, Any]] = None,
        scheduler_class: Optional[type] = None,
        scheduler_init_args: Optional[Dict[str, Any]] = None,
        scheduler_interval: Literal['epoch', 'step'] = 'epoch',
        scheduler_monitor: Optional[str] = None,
        crop_border: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        model_hparams = getattr(model, 'hparams', {})
        self._hparams = self._flatten_hparams({
            **self._hparams,
            'model': model_hparams,
            'criterion': type(criterion).__name__ if criterion is not None else 'MSELoss',
        })
        self.model = model

        if example_input_shape is not None:
            self.example_input_array = torch.zeros(1, *example_input_shape)

        # Store training hyper-parameters
        self.lr = lr
        self.layer_optim_params = layer_optim_params

        self.optimizer_class = optimizer_class
        self.optimizer_init_args = optimizer_init_args or {}
        self.scheduler_class = scheduler_class
        self.scheduler_init_args = scheduler_init_args or {}
        self.scheduler_interval = scheduler_interval
        self.scheduler_monitor = scheduler_monitor or f'val_psnr({psnr_channels[0]})'
        self.crop_border = crop_border
        self.psnr_channels = psnr_channels
        self.model_colorspace = model_colorspace

        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()

        # Validation PSNR metrics — one torchmetrics instance per tracked key
        metric_keys: List[str] = []
        for cs in psnr_channels:
            if separate_psnr:
                metric_keys.extend(self._CS_CHANNEL_NAMES[cs])
            metric_keys.append(cs)
        self.val_psnr_metrics = torch.nn.ModuleDict({
            k: torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
            for k in metric_keys
        })
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    @staticmethod
    def _flatten_hparams(hparams: Dict[str, Any], sep: str = '/') -> dict:
        """Recursively flatten nested hparams dicts/lists for clean TensorBoard HParams columns.

        None values are dropped (they add noise without aiding comparison).
        Class objects are reduced to their ``__name__``.
        """
        result = {}

        def _recurse(obj, prefix: str) -> None:
            if obj is None:
                return
            if isinstance(obj, type):
                result[prefix] = obj.__name__
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _recurse(v, f"{prefix}{sep}{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    _recurse(v, f"{prefix}{sep}{i}")
            elif isinstance(obj, (bool, int, float, str)):
                result[prefix] = obj
            else:
                result[prefix] = str(obj)

        for k, v in hparams.items():
            _recurse(v, k)

        return result

    @staticmethod
    def _rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
        """Convert a normalised RGB tensor (B, 3, H, W) to YCbCr using BT.601 full-range."""
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        y  =  0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.169 * r - 0.331 * g + 0.500 * b + 0.5
        cr =  0.500 * r - 0.419 * g - 0.081 * b + 0.5
        return torch.cat([y, cb, cr], dim=1)

    @staticmethod
    def _ycbcr_to_rgb(img: torch.Tensor) -> torch.Tensor:
        """Convert a YCbCr tensor (B, 3, H, W) to RGB using BT.601 full-range."""
        y, cb, cr = img[:, 0:1], img[:, 1:2] - 0.5, img[:, 2:3] - 0.5
        r = y + 1.402 * cr
        g = y - 0.344 * cb - 0.714 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

    def _extract_model_input(
        self, lr_img: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare model input from a full RGB LR tensor.

        Returns ``(model_input, lr_ycbcr)`` where ``lr_ycbcr`` is the full
        LR YCbCr tensor retained for channel reconstruction, or ``None`` when
        not needed (``model_colorspace='RGB'``).
        """
        if self.model_colorspace == 'RGB':
            return lr_img, None
        lr_ycbcr = self._rgb_to_ycbcr(lr_img)
        if self.model_colorspace == 'Y':
            return lr_ycbcr[:, 0:1], lr_ycbcr
        if self.model_colorspace == 'YCbCr':
            return lr_ycbcr, None
        raise ValueError(f"Unknown model_colorspace '{self.model_colorspace}'. Expected 'RGB', 'Y', or 'YCbCr'.")

    def _reconstruct_sr_rgb(
        self, sr_model: torch.Tensor, lr_ycbcr: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Reconstruct a full RGB SR image from the model output.

        For ``model_colorspace='Y'``, stitches SR-Y with bicubic Cb/Cr taken
        from the LR image (centre-cropped to match SR spatial size) before
        converting to RGB.
        """
        if self.model_colorspace == 'RGB':
            return sr_model
        if self.model_colorspace == 'Y':
            cb = torchvision.transforms.functional.center_crop(lr_ycbcr[:, 1:2], sr_model.shape[-2:])
            cr = torchvision.transforms.functional.center_crop(lr_ycbcr[:, 2:3], sr_model.shape[-2:])
            return self._ycbcr_to_rgb(torch.cat([sr_model, cb, cr], dim=1))
        if self.model_colorspace == 'YCbCr':
            return self._ycbcr_to_rgb(sr_model)
        raise ValueError(f"Unknown model_colorspace '{self.model_colorspace}'.")

    def _build_psnr_tensors(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Pre-compute (sr, hr) tensor pairs for every tracked PSNR key.

        Colorspace conversions are performed at most once per call.
        """
        keys = set(self.val_psnr_metrics.keys())
        tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        if keys & {'RGB', 'R', 'G', 'B'}:
            tensors['RGB'] = (sr, hr)
            tensors['R']   = (sr[:, 0:1], hr[:, 0:1])
            tensors['G']   = (sr[:, 1:2], hr[:, 1:2])
            tensors['B']   = (sr[:, 2:3], hr[:, 2:3])

        if keys & {'YCbCr', 'Y', 'Cb', 'Cr'}:
            sr_ycc = self._rgb_to_ycbcr(sr)
            hr_ycc = self._rgb_to_ycbcr(hr)
            tensors['YCbCr'] = (sr_ycc, hr_ycc)
            tensors['Y']     = (sr_ycc[:, 0:1], hr_ycc[:, 0:1])
            tensors['Cb']    = (sr_ycc[:, 1:2], hr_ycc[:, 1:2])
            tensors['Cr']    = (sr_ycc[:, 2:3], hr_ycc[:, 2:3])

        return tensors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — delegates to the wrapped SR model.

        Args:
            x (torch.Tensor): Low-resolution input of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Super-resolved output of shape ``(B, C, H', W')``.
        """
        return self.model(x)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Shared forward pass and loss computation used by all step methods.

        Runs the model on the low-resolution input, centre-crops the
        high-resolution target to match the SR output spatial size (needed
        when the model uses ``padding='valid'``), and computes the loss.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A ``(lr, hr)`` pair
                of image tensors with shape ``(B, C, H, W)``.

        Returns:
            Tuple containing:
                - **loss** — scalar loss tensor (computed in ``model_colorspace``).
                - **lr_img** — the low-resolution input (full RGB).
                - **hr_img** — the original high-resolution target (full RGB).
                - **sr_rgb** — the super-resolved output reconstructed to full RGB.
                - **hr_cropped** — the centre-cropped HR target (full RGB) matching
                  ``sr_rgb`` spatially.
        """
        lr_img, hr_img = batch

        # Prepare model input and retain aux YCbCr channels for reconstruction
        model_input, lr_ycbcr = self._extract_model_input(lr_img)
        sr_model = self.model(model_input)

        # Reconstruct full RGB SR image
        sr_rgb = self._reconstruct_sr_rgb(sr_model, lr_ycbcr)

        # Centre-crop HR (RGB) to match SR spatial size
        hr_cropped = torchvision.transforms.functional.center_crop(hr_img, sr_rgb.shape[-2:])

        # Loss is computed in model colorspace against HR in that same space
        if self.model_colorspace == 'Y':
            hr_for_loss = self._rgb_to_ycbcr(hr_cropped)[:, 0:1]
        elif self.model_colorspace == 'YCbCr':
            hr_for_loss = self._rgb_to_ycbcr(hr_cropped)
        else:
            hr_for_loss = hr_cropped
        loss = self.criterion(sr_model, hr_for_loss)

        return loss, lr_img, hr_img, sr_rgb, hr_cropped

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Computes the loss on the batch and logs it for both the current
        step and the epoch aggregate.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A ``(lr, hr)`` pair
                of image tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss (used by Lightning for
                backpropagation).
        """
        loss, *_ = self._step(batch)

        self.log('train_loss', loss, prog_bar=True, on_step=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Perform a single validation step.

        Computes and logs the validation loss, PSNR, and SSIM (epoch-level
        aggregation only).  Metrics are only logged for the primary
        validation dataloader (``dataloader_idx == 0``); benchmark
        dataloaders (Set5, Set14, etc.) are handled by
        :class:`~sisr.callbacks.BenchmarkImageLogger`.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A ``(lr, hr)`` pair
                of image tensors.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the current dataloader when
                multiple validation dataloaders are used.  Defaults to ``0``.
        """
        # Only log aggregate metrics for the primary validation dataloader
        if dataloader_idx != 0:
            return

        loss, lr_img, hr_img, sr, hr_cropped = self._step(batch)

        if self.crop_border > 0:
            n = self.crop_border
            sr = sr[..., n:-n, n:-n]
            hr_cropped = hr_cropped[..., n:-n, n:-n]

        psnr_tensors = self._build_psnr_tensors(sr, hr_cropped)
        ssim = self.val_ssim(sr, hr_cropped)

        # add_dataloader_idx=False keeps metric names clean (no '/dataloader_idx_0' suffix)
        # so checkpoints and schedulers that monitor these metrics work with multiple val dataloaders.
        self.log('val_loss', loss, prog_bar=True, on_step=False, add_dataloader_idx=False)
        for key, metric in self.val_psnr_metrics.items():
            sr_t, hr_t = psnr_tensors[key]
            self.log(
                f'val_psnr({key})', metric(sr_t, hr_t),
                prog_bar=(key == self.psnr_channels[0]),
                on_step=False, add_dataloader_idx=False,
            )
        self.log('val_ssim', ssim, prog_bar=True, on_step=False, add_dataloader_idx=False)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Test step for `cli test --ckpt_path <path>` final evaluation.

        All metric computation, per-image scalar logging, and image-strip
        emission for the test sets is handled by
        :class:`~sisr.training.callbacks.BenchmarkImageLogger` in its
        ``on_test_batch_end`` / ``on_test_epoch_end`` hooks.  This method
        exists so Lightning iterates ``trainer.test_dataloaders`` and the
        callback hooks fire — it deliberately does no work itself to avoid
        a redundant forward pass.
        """
        return None

    def configure_optimizers(self):
        """
        Configure the optimizer and (optionally) the LR scheduler.

        Builds per-Conv2d parameter groups so that each layer can have its
        own learning rate when ``layer_optim_params`` is provided.  If a
        scheduler is configured it is returned alongside the optimizer; for
        ``ReduceLROnPlateau`` the monitored metric is included automatically.

        Returns:
            Either a bare optimizer or a dict with ``'optimizer'`` and
            ``'lr_scheduler'`` keys as expected by Lightning.

        Raises:
            ValueError: If ``layer_optim_params`` length does not match the
                number of Conv2d layers in the model.
        """
        # Collect all Conv2d layers in the model
        conv_layers = [
            m for m in self.model.modules()
            if isinstance(m, torch.nn.Conv2d)
        ]

        # Fall back to uniform base LR when no per-layer config is given
        layer_optim_params = self.layer_optim_params if self.layer_optim_params is not None else [
            {'lr': self.lr}] * len(conv_layers)

        if len(conv_layers) != len(layer_optim_params):
            raise ValueError(
                "layer_optim_params length must match number of Conv2d layers")

        # Build per-layer parameter groups
        param_groups = [
            {'params': layer.parameters(), **layer_optim_param}
            for layer, layer_optim_param in zip(conv_layers, layer_optim_params)
        ]

        optimizer = self.optimizer_class(
            param_groups,
            **self.optimizer_init_args,
        )

        if self.scheduler_class is None:
            return optimizer

        scheduler = self.scheduler_class(
            optimizer,
            **self.scheduler_init_args,
        )

        # ReduceLROnPlateau requires a monitored metric
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.scheduler_monitor,
                    'interval': self.scheduler_interval,
                },
            }

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': self.scheduler_interval,
            },
        }
