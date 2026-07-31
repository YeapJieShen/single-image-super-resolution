"""Generic Lightning wrapper for SR models — :class:`SRLightning`.

The Lightning module is architecture-agnostic: per-paper behavior lives
in :class:`~sisr.training.config.SRTrainingConfig` /
:class:`~sisr.training.config.SREvalConfig` subclasses (e.g.
:class:`~sisr.models.srcnn.SRCNNTrainingConfig`), not in the module
itself. Optimizer / LR scheduler are wired in from top-level YAML keys by
:class:`~sisr.cli.SRLightningCLI`.
"""

import dataclasses
from typing import Any

import lightning
import torch
import torchmetrics.functional
import torchvision
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ..colorspace import rgb_to_ycbcr
from ..models.base import SRModel
from ..processors import SRProcessor
from .config import SREvalConfig, SRTrainingConfig


class SRLightning(lightning.LightningModule):
    """Generic Lightning wrapper for single-image super-resolution models.

    Composes an :class:`~sisr.models.base.SRModel` with an
    :class:`~sisr.processors.SRProcessor`. The model is a pure tensor function;
    the processor handles per-batch colorspace conversion between the dataset's
    RGB format and the model's input/output colorspace. Both are wired
    independently in YAML via ``class_path``.

    Args:
        model: An initialised :class:`SRModel` subclass (e.g. :class:`SRCNN`,
            :class:`SRResNet`). Required.
        processor: An :class:`SRProcessor` subclass (e.g.
            :class:`~sisr.processors.RGBProcessor`,
            :class:`~sisr.processors.YChannelProcessor`,
            :class:`~sisr.processors.YCbCrProcessor`). Required.
        training_config: Per-architecture training settings (layer_lrs,
            example_input_shape, init_*). Defaults to base
            :class:`SRTrainingConfig` (uniform LR, no paper init).
        eval_config: Per-architecture evaluation settings (crop_border,
            psnr_channels, separate_psnr). Defaults to base :class:`SREvalConfig`.
        criterion: Loss instance. Defaults to :class:`torch.nn.MSELoss`.
        optimizer: ``OptimizerCallable`` populated from top-level YAML
            ``optimizer:``. Defaults to :class:`torch.optim.Adam`.
        lr_scheduler: ``LRSchedulerCallable`` or ``None``. Defaults to ``None``.

    Raises:
        TypeError: If ``model`` is not an :class:`SRModel` subclass, or
            if ``processor`` is not an :class:`SRProcessor` subclass.
    """

    def __init__(
        self,
        model: SRModel,
        processor: SRProcessor,
        training_config: SRTrainingConfig | None = None,
        eval_config: SREvalConfig | None = None,
        criterion: torch.nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ):
        super().__init__()
        if not isinstance(model, SRModel):
            raise TypeError(
                f"SRLightning requires an SRModel subclass; got "
                f"{type(model).__name__}. Update your YAML to point at a "
                f"class under sisr.models.* that inherits from SRModel."
            )
        if not isinstance(processor, SRProcessor):
            raise TypeError(
                f"SRLightning requires an SRProcessor subclass; got "
                f"{type(processor).__name__}. Update your YAML to point "
                f"at a class under sisr.processors.* that inherits from "
                f"SRProcessor."
            )

        self.save_hyperparameters(
            ignore=["model", "processor", "criterion", "optimizer", "lr_scheduler"]
        )

        self.model = model
        self.processor = processor
        self.training_config = training_config or SRTrainingConfig()
        self.eval_config = eval_config or SREvalConfig()
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.training_config.validate_against(self.model, self.processor)

        if self.training_config.init_strategy == "paper":
            model.reset_parameters(
                mean=self.training_config.init_mean,
                std=self.training_config.init_std,
            )

        # Merge model.hparams + processor identity into Lightning hparams for TensorBoard HParams.
        # Configs are expanded via dataclasses.asdict so each field becomes its own HParams column.
        self._hparams = self._flatten_hparams(
            {
                **self._hparams,
                "training_config": dataclasses.asdict(self.training_config),
                "eval_config": dataclasses.asdict(self.eval_config),
                "model": model.hparams,
                "processor": type(processor).__name__,
                "criterion": type(self.criterion).__name__,
            }
        )

        if self.training_config.example_input_shape is not None:
            self.example_input_array = torch.zeros(1, *self.training_config.example_input_shape)

    @staticmethod
    def _flatten_hparams(hparams: dict[str, Any], sep: str = "/") -> dict:
        """Recursively flatten nested hparams dicts/lists for clean TensorBoard HParams columns.

        None values are dropped (they add noise without aiding comparison).
        Class objects are reduced to their ``__name__``.

        Args:
            hparams: Nested mapping of hparams (may contain dicts, lists,
                tuples, scalars, or class objects).
            sep: Separator used to join nested keys. Defaults to ``'/'``.

        Returns:
            Flat dict with ``sep``-joined keys and scalar leaves.
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
            elif isinstance(obj, list | tuple):
                for i, v in enumerate(obj):
                    _recurse(v, f"{prefix}{sep}{i}")
            elif isinstance(obj, bool | int | float | str):
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

        Args:
            sr: SR tensor of shape ``(B, 3, H, W)`` in RGB.
            hr: HR tensor of shape ``(B, 3, H, W)`` in RGB.

        Returns:
            Mapping from PSNR key (``'RGB'``, ``'Y'``, ``'Cb'``, ...) to
            ``(sr_subset, hr_subset)`` tensor pair ready for PSNR
            computation.
        """
        keys = set(self.eval_config.psnr_keys)
        tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        if keys & {"RGB", "R", "G", "B"}:
            tensors["RGB"] = (sr, hr)
            tensors["R"] = (sr[:, 0:1], hr[:, 0:1])
            tensors["G"] = (sr[:, 1:2], hr[:, 1:2])
            tensors["B"] = (sr[:, 2:3], hr[:, 2:3])

        if keys & {"YCbCr", "Y", "Cb", "Cr"}:
            sr_ycc = rgb_to_ycbcr(sr)
            hr_ycc = rgb_to_ycbcr(hr)
            tensors["YCbCr"] = (sr_ycc, hr_ycc)
            tensors["Y"] = (sr_ycc[:, 0:1], hr_ycc[:, 0:1])
            tensors["Cb"] = (sr_ycc[:, 1:2], hr_ycc[:, 1:2])
            tensors["Cr"] = (sr_ycc[:, 2:3], hr_ycc[:, 2:3])

        return tensors

    @staticmethod
    def _mean_psnr(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Mean of the per-image PSNRs across the batch (SR-standard reduction).

        Scores each image independently (``dim=(1, 2, 3)``) before the batch
        mean, so the result is invariant to the val ``batch_size``. This differs
        from pooling the whole batch into one PSNR — the deviation a stateful
        ``PeakSignalNoiseRatio`` with default ``dim`` would introduce.

        Args:
            sr: SR tensor of shape ``(B, C, H, W)`` in ``[0, 1]``.
            hr: HR tensor of the same shape.

        Returns:
            Scalar tensor — the batch mean of the per-image PSNRs.
        """
        return torchmetrics.functional.image.peak_signal_noise_ratio(
            sr, hr, data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapped SR model on ``x`` and return its raw output.

        Pure inference path — no colorspace conversion, no metrics, no
        cropping. Used by ``trainer.predict`` and direct ``module(x)``
        calls. The training / validation paths go through :meth:`_step`,
        which adds the colorspace pipeline.

        Args:
            x: LR input tensor of shape ``(B, C, H, W)``.

        Returns:
            SR tensor as produced by the wrapped model. Shape depends on
            the architecture (same as ``x`` for SRCNN; ``(B, C, H*scale,
            W*scale)`` for SRResNet).
        """
        return self.model(x)

    def _forward_lr(self, lr_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LR-only core of the forward pipeline: extract → model → reconstruct.

        Factored out of :meth:`_forward_sr` so :meth:`predict_step` can share
        the exact colorspace pipeline instead of forking it — HR is only ever
        needed for the center-crop :meth:`_forward_sr` adds on top, which has
        no meaning without an HR reference at inference time.

        Args:
            lr_img: LR batch, RGB ``float32`` in ``[0, 1]``, shape
                ``(B, 3, H, W)``.

        Returns:
            ``(sr_model_out, sr_rgb)`` — the raw model output in the model IO
            colorspace, and the reconstructed SR RGB.
        """
        model_input = self.processor.extract(lr_img)
        sr_model_out = self.model(model_input)
        sr_rgb = self.processor.reconstruct(sr_model_out, lr_img)
        return sr_model_out, sr_rgb

    def _forward_sr(
        self, lr_img: torch.Tensor, hr_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Canonical SR forward: extract → model → reconstruct → crop HR.

        The single source of truth for the forward pipeline. Shared by
        :meth:`_step` (which also needs the model-space output for the loss),
        :meth:`predict_rgb` (the public scoring seam), and — via
        :meth:`_forward_lr` — :meth:`predict_step` (the HR-free inference
        seam), so the training, benchmark-logging, and prediction paths
        cannot silently diverge.

        Args:
            lr_img: LR batch, RGB ``float32`` in ``[0, 1]``, shape
                ``(B, 3, H, W)``.
            hr_img: HR batch, RGB ``float32`` in ``[0, 1]``.

        Returns:
            ``(sr_model_out, sr_rgb, hr_cropped)`` — the raw model output in
            the model IO colorspace, the reconstructed SR RGB, and HR
            center-cropped to the SR spatial size.
        """
        sr_model_out, sr_rgb = self._forward_lr(lr_img)
        hr_cropped = torchvision.transforms.functional.center_crop(hr_img, sr_rgb.shape[-2:])
        return sr_model_out, sr_rgb, hr_cropped

    def predict_rgb(
        self, lr_img: torch.Tensor, hr_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the SR forward and return spatially-aligned SR / HR RGB tensors.

        Public scoring seam consumed by the training path (via :meth:`_step`)
        and by :class:`~sisr.training.callbacks.BenchmarkImageLogger`, so
        neither re-implements the ``extract → model → reconstruct →
        center_crop`` pipeline nor reaches into ``.model`` / ``.processor``.
        Gradient tracking follows the caller's context — wrap in
        ``torch.no_grad()`` for pure inference.

        Args:
            lr_img: LR batch, RGB ``float32`` in ``[0, 1]``, shape
                ``(B, 3, H, W)``.
            hr_img: HR batch, RGB ``float32`` in ``[0, 1]``.

        Returns:
            ``(sr_rgb, hr_cropped)`` — SR RGB and HR center-cropped to the SR
            spatial size, both RGB ``float32``.
        """
        _, sr_rgb, hr_cropped = self._forward_sr(lr_img, hr_img)
        return sr_rgb, hr_cropped

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared forward + loss for training and validation steps.

        The processor handles all colorspace conversion; loss is computed in
        the model's IO colorspace (against HR converted to the same space
        via ``processor.extract``). Metrics downstream consume ``sr_rgb`` /
        ``hr_cropped`` (both full RGB, spatially aligned).

        Args:
            batch: ``(lr_img, hr_img)`` tuple from a loader. Both RGB,
                ``float32`` in ``[0, 1]``.

        Returns:
            ``(loss, lr_img, hr_img, sr_rgb, hr_cropped)``. ``loss`` is a
            scalar tensor; ``sr_rgb`` and ``hr_cropped`` are RGB tensors
            with matching spatial size.
        """
        lr_img, hr_img = batch

        sr_model_out, sr_rgb, hr_cropped = self._forward_sr(lr_img, hr_img)
        hr_for_loss = self.processor.extract(hr_cropped)
        loss = self.criterion(sr_model_out, hr_for_loss)

        return loss, lr_img, hr_img, sr_rgb, hr_cropped

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute training loss for one batch and log it.

        Delegates the forward + colorspace + loss pipeline to :meth:`_step`
        and logs ``train_loss`` on every step for the progress bar.

        Args:
            batch: ``(lr_img, hr_img)`` tuple as produced by the
                :class:`SRDataModule` train loader. Both are ``float32`` in
                ``[0, 1]``.
            batch_idx: Index of the batch within the current epoch.

        Returns:
            Scalar loss tensor for the optimizer.
        """
        loss, *_ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute and log validation metrics for the *primary* val loader (idx 0).

        Benchmark / test loaders (idx >= 1) are handled by
        :class:`~sisr.training.callbacks.BenchmarkImageLogger`.

        Args:
            batch: ``(lr_img, hr_img)`` tuple from the val loader.
            batch_idx: Index of the batch within the current val epoch.
            dataloader_idx: Index of the loader within the list returned
                by :meth:`SRDataModule.val_dataloader`. ``0`` is the
                primary val set; anything else is skipped.
        """
        if dataloader_idx != 0:
            return

        loss, lr_img, hr_img, sr, hr_cropped = self._step(batch)

        n = self.eval_config.crop_border
        if n > 0:
            sr = sr[..., n:-n, n:-n]
            hr_cropped = hr_cropped[..., n:-n, n:-n]

        psnr_tensors = self._build_psnr_tensors(sr, hr_cropped)

        # add_dataloader_idx=False keeps metric names clean — needed because the
        # primary val loader is at idx 0 of a list that also contains test loaders.
        self.log("val_loss", loss, prog_bar=True, on_step=False, add_dataloader_idx=False)
        primary = self.eval_config.psnr_channels[0]
        for key in self.eval_config.psnr_keys:
            sr_t, hr_t = psnr_tensors[key]
            self.log(
                f"val_psnr({key})",
                self._mean_psnr(sr_t, hr_t),
                prog_bar=(key == primary),
                on_step=False,
                add_dataloader_idx=False,
            )
        self.log(
            "val_ssim",
            torchmetrics.functional.image.structural_similarity_index_measure(
                sr, hr_cropped, data_range=1.0
            ),
            prog_bar=True,
            on_step=False,
            add_dataloader_idx=False,
        )

    def on_train_start(self) -> None:
        """Register val metric tags with TensorBoard's HParams tab.

        Replaces Lightning's default ``hp_metric`` placeholder (disabled via
        ``default_hp_metric=False`` on the logger) with the real validation
        metrics this module emits. Placeholder values are ``0.0``; TensorBoard's
        HParams plugin then fills the columns from the matching scalar tags
        logged by :meth:`validation_step`.
        """
        tb_loggers = [
            logger
            for logger in self.loggers
            if isinstance(logger, lightning.pytorch.loggers.TensorBoardLogger)
        ]
        if not tb_loggers:
            return
        metrics = {
            **{f"val_psnr({k})": 0.0 for k in self.eval_config.psnr_keys},
            "val_ssim": 0.0,
        }
        for tb in tb_loggers:
            tb.log_hyperparams(self.hparams, metrics)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op so Lightning iterates ``trainer.test_dataloaders``.

        All metric computation, per-image logging, and image-strip
        emission for the test sets happens in
        :class:`~sisr.training.callbacks.BenchmarkImageLogger.on_test_*`.

        Args:
            batch: ``(lr_img, hr_img)`` tuple from the test loader
                (unused).
            batch_idx: Index of the batch within the current test epoch
                (unused).
            dataloader_idx: Index of the test loader (unused).
        """
        return None

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Run the HR-free inference pipeline: extract → model → reconstruct.

        Shares :meth:`_forward_lr` with :meth:`_forward_sr` (the pipeline
        backing :meth:`_step` / :meth:`predict_rgb`) — the same colorspace
        pipeline minus the HR-dependent center-crop and scoring, neither of
        which has meaning without an HR reference. Paired with
        :meth:`~sisr.training.SRDataModule.predict_dataloader` and typically
        consumed by :class:`~sisr.training.callbacks.SRPredictionWriter`.

        Args:
            batch: LR batch, RGB ``float32`` in ``[0, 1]``, shape
                ``(B, 3, H, W)``, as produced by
                :class:`~sisr.datasets.predict.PredictDataset`.
            batch_idx: Index of the batch within the predict run (unused).
            dataloader_idx: Index of the predict loader (unused — a single
                predict loader is expected).

        Returns:
            SR RGB tensor, ``float32`` in ``[0, 1]``, shape
            ``(B, 3, H', W')`` — same size as the input for SRCNN,
            ``H'=H*scale``/``W'=W*scale`` for SRResNet.
        """
        _, sr_rgb = self._forward_lr(batch)
        return sr_rgb

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build optimizer (and optional scheduler) from top-level YAML.

        Uniform LR by default — ``self.optimizer(self.parameters())``
        exactly matches what LightningCLI's ``auto_configure_optimizers``
        would do.

        When ``training_config.layer_lrs`` is set, builds per-``Conv2d``
        ``param_groups`` with explicit absolute LRs. This requires every
        trainable parameter to live inside a ``Conv2d`` (SRCNN-style — no
        BatchNorm / PReLU); the validation below makes that explicit.

        Returns:
            The constructed optimizer, or a ``([optimizer], [scheduler])``
            tuple if ``self.lr_scheduler`` is set. Lightning accepts both.

        Raises:
            ValueError: If ``training_config.layer_lrs`` length does not
                match the model's ``Conv2d`` count, or if any trainable
                parameter lives outside a ``Conv2d`` while ``layer_lrs``
                is set.
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
                n
                for n, p in self.model.named_parameters()
                if p.requires_grad and id(p) not in conv_params
            ]
            if other:
                raise ValueError(
                    f"training_config.layer_lrs requires every trainable param to live in "
                    f"a Conv2d; these do not: {other}. Set layer_lrs=None for models with "
                    f"non-Conv layers (BatchNorm, PReLU, etc.)."
                )
            param_groups = [
                {"params": list(layer.parameters()), "lr": lr}
                for layer, lr in zip(conv_layers, lrs, strict=False)
            ]
            optimizer = self.optimizer(param_groups)

        if self.lr_scheduler is None:
            return optimizer
        scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [scheduler]
