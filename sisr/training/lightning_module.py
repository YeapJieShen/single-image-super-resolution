import torch
import torchvision
import torchmetrics
import lightning
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from typing import Any

from ..utils import (
    extract_model_input,
    reconstruct_sr_rgb,
    rgb_to_ycbcr,
)
from .config import SREvalConfig, SRTrainingConfig


class SRLightning(lightning.LightningModule):
    """
    Generic Lightning wrapper for single-image super-resolution models.

    Architecture-agnostic: any ``torch.nn.Module`` that accepts an LR tensor
    of shape ``(B, C, H, W)`` and returns an SR tensor can be plugged in.
    The wrapped model is passed in pre-instantiated (jsonargparse / LightningCLI
    handle the construction via ``class_path`` / ``init_args``).

    Per-architecture training and evaluation knobs live in
    :class:`~sisr.training.config.SRTrainingConfig` and
    :class:`~sisr.training.config.SREvalConfig` (and their subclasses next to
    each model, e.g. :class:`~sisr.models.srcnn.SRCNNTrainingConfig`).
    Optimizer and LR scheduler are wired from top-level YAML keys (``optimizer:``,
    ``lr_scheduler:``) by :class:`~sisr.cli.SRLightningCLI` via
    ``link_arguments``; the user does not pass them explicitly under
    ``model.init_args``.

    Args:
        model: An initialised SR model instance.  If it exposes an
            ``hparams`` mapping (as :class:`~sisr.models.srcnn.SRCNN` does),
            those values are merged into the Lightning hparams so model
            specs appear alongside training params in TensorBoard HParams.
        training_config: Per-architecture training settings (model_colorspace,
            layer_lrs, example_input_shape).  Defaults to
            :class:`SRTrainingConfig`'s base defaults (RGB, uniform LR).
        eval_config: Per-architecture evaluation settings (crop_border,
            psnr_channels, separate_psnr).  Defaults to
            :class:`SREvalConfig`'s base defaults (no crop, RGB-only PSNR).
        criterion: Loss instance.  Defaults to :class:`torch.nn.MSELoss`
            when ``None``.
        optimizer: ``OptimizerCallable`` (``functools.partial(optimizer_cls,
            **init_args)``-style) — populated from top-level YAML
            ``optimizer:``.  Defaults to :class:`torch.optim.Adam`.
        lr_scheduler: ``LRSchedulerCallable`` or ``None`` — populated from
            top-level YAML ``lr_scheduler:``.  Defaults to ``None``
            (constant LR).
    """

    _CS_CHANNEL_NAMES: dict[str, tuple[str, ...]] = {
        'RGB':   ('R', 'G', 'B'),
        'YCbCr': ('Y', 'Cb', 'Cr'),
    }

    def __init__(
        self,
        model: torch.nn.Module,
        training_config: SRTrainingConfig | None = None,
        eval_config: SREvalConfig | None = None,
        criterion: torch.nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion', 'optimizer', 'lr_scheduler'])

        self.model = model
        self.training_config = training_config or SRTrainingConfig()
        self.eval_config = eval_config or SREvalConfig()
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        strategy = getattr(self.training_config, 'init_strategy', 'default')
        if strategy == 'paper' and hasattr(model, 'reset_parameters'):
            mean = getattr(self.training_config, 'init_mean', 0.0)
            std = getattr(self.training_config, 'init_std', 0.01)
            model.reset_parameters(mean=mean, std=std)

        # Merge model.hparams into Lightning hparams for TensorBoard HParams.
        model_hparams = getattr(model, 'hparams', {})
        self._hparams = self._flatten_hparams({
            **self._hparams,
            'model': model_hparams,
            'criterion': type(self.criterion).__name__,
        })

        if self.training_config.example_input_shape is not None:
            self.example_input_array = torch.zeros(1, *self.training_config.example_input_shape)

        metric_keys: list[str] = []
        for cs in self.eval_config.psnr_channels:
            if self.eval_config.separate_psnr:
                metric_keys.extend(self._CS_CHANNEL_NAMES[cs])
            metric_keys.append(cs)
        self.val_psnr_metrics = torch.nn.ModuleDict({
            k: torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
            for k in metric_keys
        })
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    @staticmethod
    def _flatten_hparams(hparams: dict[str, Any], sep: str = '/') -> dict:
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

    def _build_psnr_tensors(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Pre-compute (sr, hr) tensor pairs for every tracked PSNR key.

        Colorspace conversions are performed at most once per call.
        """
        keys = set(self.val_psnr_metrics.keys())
        tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        if keys & {'RGB', 'R', 'G', 'B'}:
            tensors['RGB'] = (sr, hr)
            tensors['R']   = (sr[:, 0:1], hr[:, 0:1])
            tensors['G']   = (sr[:, 1:2], hr[:, 1:2])
            tensors['B']   = (sr[:, 2:3], hr[:, 2:3])

        if keys & {'YCbCr', 'Y', 'Cb', 'Cr'}:
            sr_ycc = rgb_to_ycbcr(sr)
            hr_ycc = rgb_to_ycbcr(hr)
            tensors['YCbCr'] = (sr_ycc, hr_ycc)
            tensors['Y']     = (sr_ycc[:, 0:1], hr_ycc[:, 0:1])
            tensors['Cb']    = (sr_ycc[:, 1:2], hr_ycc[:, 1:2])
            tensors['Cr']    = (sr_ycc[:, 2:3], hr_ycc[:, 2:3])

        return tensors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Shared forward + loss for training and validation steps.

        Returns ``(loss, lr_img, hr_img, sr_rgb, hr_cropped)``.  Loss is
        computed in :attr:`SRTrainingConfig.model_colorspace`; metrics
        downstream consume ``sr_rgb`` / ``hr_cropped`` (both full RGB,
        spatially aligned).
        """
        lr_img, hr_img = batch
        cs = self.training_config.model_colorspace

        model_input, lr_ycbcr = extract_model_input(lr_img, cs)
        sr_model = self.model(model_input)
        sr_rgb = reconstruct_sr_rgb(sr_model, lr_ycbcr, cs)

        hr_cropped = torchvision.transforms.functional.center_crop(hr_img, sr_rgb.shape[-2:])

        # Loss is computed in model colorspace against HR in that same space.
        if cs == 'Y':
            hr_for_loss = rgb_to_ycbcr(hr_cropped)[:, 0:1]
        elif cs == 'YCbCr':
            hr_for_loss = rgb_to_ycbcr(hr_cropped)
        else:
            hr_for_loss = hr_cropped
        loss = self.criterion(sr_model, hr_for_loss)

        return loss, lr_img, hr_img, sr_rgb, hr_cropped

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, *_ = self._step(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute and log validation metrics for the *primary* val loader (idx 0).

        Benchmark / test loaders (idx >= 1) are handled by
        :class:`~sisr.training.callbacks.BenchmarkImageLogger`.
        """
        if dataloader_idx != 0:
            return

        loss, lr_img, hr_img, sr, hr_cropped = self._step(batch)

        n = self.eval_config.crop_border
        if n > 0:
            sr = sr[..., n:-n, n:-n]
            hr_cropped = hr_cropped[..., n:-n, n:-n]

        psnr_tensors = self._build_psnr_tensors(sr, hr_cropped)
        ssim = self.val_ssim(sr, hr_cropped)

        # add_dataloader_idx=False keeps metric names clean — needed because the
        # primary val loader is at idx 0 of a list that also contains test loaders.
        self.log('val_loss', loss, prog_bar=True, on_step=False, add_dataloader_idx=False)
        primary = self.eval_config.psnr_channels[0]
        for key, metric in self.val_psnr_metrics.items():
            sr_t, hr_t = psnr_tensors[key]
            self.log(
                f'val_psnr({key})', metric(sr_t, hr_t),
                prog_bar=(key == primary),
                on_step=False, add_dataloader_idx=False,
            )
        self.log('val_ssim', ssim, prog_bar=True, on_step=False, add_dataloader_idx=False)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op so Lightning iterates ``trainer.test_dataloaders``.

        All metric computation, per-image logging, and image-strip emission
        for the test sets happens in
        :class:`~sisr.training.callbacks.BenchmarkImageLogger.on_test_*`.
        """
        return None

    def configure_optimizers(self):
        """Build optimizer (and optional scheduler) from top-level YAML.

        Uniform LR by default — ``self.optimizer(self.parameters())`` exactly
        matches what LightningCLI's ``auto_configure_optimizers`` would do.

        When ``training_config.layer_lrs`` is set, builds per-``Conv2d``
        ``param_groups`` with explicit absolute LRs.  This requires every
        trainable parameter to live inside a ``Conv2d`` (SRCNN-style — no
        BatchNorm / PReLU); the validation below makes that explicit.
        """
        lrs = self.training_config.layer_lrs
        if lrs is None:
            optimizer = self.optimizer(self.parameters())
        else:
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            if len(conv_layers) != len(lrs):
                raise ValueError(
                    f"training_config.layer_lrs length {len(lrs)} != "
                    f"Conv2d count {len(conv_layers)}"
                )
            conv_params = {id(p) for layer in conv_layers for p in layer.parameters()}
            other = [
                n for n, p in self.model.named_parameters()
                if p.requires_grad and id(p) not in conv_params
            ]
            if other:
                raise ValueError(
                    f"training_config.layer_lrs requires every trainable param to live in "
                    f"a Conv2d; these do not: {other}. Set layer_lrs=None for models with "
                    f"non-Conv layers (BatchNorm, PReLU, etc.)."
                )
            param_groups = [
                {'params': list(layer.parameters()), 'lr': lr}
                for layer, lr in zip(conv_layers, lrs)
            ]
            optimizer = self.optimizer(param_groups)

        if self.lr_scheduler is None:
            return optimizer
        scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [scheduler]
