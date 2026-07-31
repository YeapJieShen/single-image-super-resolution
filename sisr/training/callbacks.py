"""Lightning callbacks for SR training.

:class:`BenchmarkImageLogger` logs per-image bicubic|SR|HR strips and PSNR/SSIM
for held-out test sets (Set5, Set14, ...) during both ``cli fit`` (every
N val cycles) and ``cli test`` (one-shot final eval).
:class:`GradNormLogger` and :class:`WeightHistogramLogger` log diagnostic
training signals; :class:`SRCheckpoint` is a thin
:class:`~lightning.pytorch.callbacks.ModelCheckpoint` preset for SR metrics.
"""

import math
from typing import Any

import lightning
import torch
import torch.nn.functional
import torchmetrics.functional
import torchvision
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class BenchmarkImageLogger(Callback):
    """Log per-image bicubic|SR|HR composites and scalar PSNR/SSIM to TensorBoard
    for one or more held-out test/benchmark dataloaders (Set5, Set14, …).

    Fires during *both* validation and test stages:

    * **During `cli fit` / `cli validate`** — `SRDataModule.val_dataloader()`
      returns ``[primary_val] + [test_loaders...]``; this callback ignores
      ``dataloader_idx == 0`` (handled by `SRLightning.validation_step`) and
      processes indices ``1..N`` against ``dataset_names``.  Images are
      logged every *n*-th val run (controlled by ``log_every_n_val_runs``)
      so TensorBoard storage stays bounded over long training schedules.
    * **During `cli test`** — `SRDataModule.test_dataloader()` returns the
      test loaders only (no primary), so indices ``0..N-1`` map straight to
      ``dataset_names``.  Images are logged every test run (no throttle —
      ``cli test`` is typically a one-shot final-eval invocation).

    Each image lands as a horizontal **bicubic | SR | HR** strip at HR scale:
    the LR is bicubic-upsampled to HR size (the standard SR baseline), and the
    SR output is center-padded when smaller than HR (``padding='valid'``) so
    all three panels share the same spatial size.

    Per-image PSNR/SSIM scalars go under ``"{name}_psnr/{filename}"`` and
    ``"{name}_ssim/{filename}"``; per-set means under
    ``"{name}_psnr({key})"`` and ``"{name}_ssim"``.

    Border cropping is sourced from ``pl_module.eval_config.crop_border`` at
    the use site; this avoids dual-knob configuration drift.

    Args:
        dataset_names: Ordered list of test set names (e.g.
            ``["Set5", "Set14"]``) matching the order of
            ``SRDataModule.test_dataloader()`` / the trailing entries of
            ``val_dataloader()``.  When ``None`` the callback auto-discovers
            from ``trainer.datamodule.test_names``.
        log_every_n_val_runs: Image-strip throttle for the val stage.  With
            step-based training (e.g. ``val_check_interval=1000``,
            ``max_steps=100_000``) val fires ~100 times, so logging every 5
            runs gives ~20 image snapshots — enough to track visual
            progress without flooding TensorBoard storage.  Default ``5``.
    """

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        log_every_n_val_runs: int = 5,
    ):
        super().__init__()
        self.dataset_names = list(dataset_names) if dataset_names else None
        self.log_every_n_val_runs = log_every_n_val_runs

        # Val: primary val is at idx 0, test sets at 1..N → {1: 'Set5', ...}
        # Test: only test sets present, at idx 0..N-1     → {0: 'Set5', ...}
        self._val_mapping: dict[int, str] = {}
        self._test_mapping: dict[int, str] = {}

        self._val_run_count = 0

        self._buffer: dict[str, list[tuple]] = {}

    def setup(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage: str,
    ):
        """Resolve ``dataset_names`` and build both dataloader_idx mappings.

        Auto-discovers ``dataset_names`` from ``trainer.datamodule.test_names``
        when not supplied at construction time.

        Args:
            trainer (lightning.Trainer): The active trainer.
            pl_module (lightning.LightningModule): The model being trained
                (unused).
            stage (str): Lightning trainer stage (unused — both val and
                test mappings are built up-front).
        """
        if self.dataset_names is None:
            dm = getattr(trainer, "datamodule", None)
            names = getattr(dm, "test_names", None) if dm is not None else None
            self.dataset_names = list(names or [])
        self._val_mapping = {i + 1: name for i, name in enumerate(self.dataset_names)}
        self._test_mapping = {i: name for i, name in enumerate(self.dataset_names)}

    def on_validation_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ):
        """Clear buffers and bump the val-run counter.

        Args:
            trainer (lightning.Trainer): The active trainer (unused).
            pl_module (lightning.LightningModule): The model being
                evaluated (unused).
        """
        self._val_run_count += 1
        self._buffer = {name: [] for name in self.dataset_names}

    def on_validation_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Collect LR/SR/HR + per-image metrics for one test loader's batch.

        Only runs for ``dataloader_idx`` in ``_val_mapping`` (i.e. test
        sets, not the primary val loader at idx 0).

        Args:
            trainer (lightning.Trainer): The active trainer.
            pl_module (lightning.LightningModule): The model being
                evaluated.
            outputs (Any): The value returned by ``validation_step``
                (unused — :class:`~sisr.training.SRLightning.validation_step`
                returns ``None`` for idx >= 1).
            batch (Any): ``(lr_img, hr_img)`` tuple from the test loader.
            batch_idx (int): Index of the batch within the current
                dataloader.
            dataloader_idx (int): Index of the loader within
                ``trainer.val_dataloaders``.
        """
        if dataloader_idx not in self._val_mapping:
            return
        self._collect_batch(
            trainer=trainer,
            pl_module=pl_module,
            batch=batch,
            batch_idx=batch_idx,
            dataset_name=self._val_mapping[dataloader_idx],
            source_dataloaders=trainer.val_dataloaders,
            dataloader_idx=dataloader_idx,
            should_log_images=self._on_image_log_interval(),
        )

    def on_validation_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ):
        """Log per-set means every val epoch; image strips every N val runs.

        Args:
            trainer (lightning.Trainer): The active trainer.
            pl_module (lightning.LightningModule): The model being
                evaluated.
        """
        self._flush_buffer(
            trainer=trainer,
            pl_module=pl_module,
            should_log_images=self._on_image_log_interval(),
        )

    def on_test_epoch_start(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """Clear buffers ahead of a test run.

        Args:
            trainer (lightning.Trainer): The active trainer (unused).
            pl_module (lightning.LightningModule): The model being
                evaluated (unused).
        """
        self._buffer = {name: [] for name in self.dataset_names}

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Collect LR/SR/HR + per-image metrics for one test loader's batch.

        ``cli test`` runs the test loaders only, so all dataloader indices
        are test sets (no primary loader at idx 0).

        Args:
            trainer (lightning.Trainer): The active trainer.
            pl_module (lightning.LightningModule): The model being
                evaluated.
            outputs (Any): The value returned by ``test_step`` (unused —
                :class:`~sisr.training.SRLightning.test_step` returns
                ``None``).
            batch (Any): ``(lr_img, hr_img)`` tuple from the test loader.
            batch_idx (int): Index of the batch within the current
                dataloader.
            dataloader_idx (int): Index of the loader within
                ``trainer.test_dataloaders``.
        """
        if dataloader_idx not in self._test_mapping:
            return
        self._collect_batch(
            trainer=trainer,
            pl_module=pl_module,
            batch=batch,
            batch_idx=batch_idx,
            dataset_name=self._test_mapping[dataloader_idx],
            source_dataloaders=trainer.test_dataloaders,
            dataloader_idx=dataloader_idx,
            should_log_images=True,
        )

    def on_test_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """Log per-set means and image strips at the end of ``cli test``.

        Args:
            trainer (lightning.Trainer): The active trainer.
            pl_module (lightning.LightningModule): The model being
                evaluated.
        """
        self._flush_buffer(
            trainer=trainer,
            pl_module=pl_module,
            should_log_images=True,
        )

    def _on_image_log_interval(self) -> bool:
        """Return True when this val run should also log image strips."""
        return self._val_run_count % self.log_every_n_val_runs == 0

    def _collect_batch(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        batch: Any,
        batch_idx: int,
        dataset_name: str,
        source_dataloaders,
        dataloader_idx: int,
        should_log_images: bool,
    ):
        """Forward one batch, compute per-image metrics, and buffer the results.

        Shared between :meth:`on_validation_batch_end` and
        :meth:`on_test_batch_end`. Border cropping is sourced from
        ``pl_module.eval_config.crop_border`` at the use site.

        Args:
            trainer (lightning.Trainer): The active trainer.
            pl_module (lightning.LightningModule): The model being evaluated.
            batch (Any): ``(lr_img, hr_img)`` tuple from the loader.
            batch_idx (int): Index of the batch within the current
                dataloader.
            dataset_name (str): Name of the test set this batch comes from.
            source_dataloaders: ``trainer.val_dataloaders`` or
                ``trainer.test_dataloaders`` — used to recover the
                underlying dataset for filename resolution.
            dataloader_idx (int): Index into ``source_dataloaders``.
            should_log_images (bool): When True the LR/SR/HR tensors are
                cached for image-strip emission in :meth:`_flush_buffer`;
                otherwise only scalar metrics are buffered.
        """
        lr_img, hr_img = batch

        with torch.no_grad():
            sr, hr_cropped = pl_module.predict_rgb(lr_img, hr_img)

        # Resolve filenames from the underlying dataset for use as TB tags.
        dataset = source_dataloaders[dataloader_idx].dataset
        batch_size = lr_img.size(0)

        for i in range(batch_size):
            global_idx = batch_idx * batch_size + i
            filename = dataset.img_paths[global_idx].stem

            sr_4d = sr[i].unsqueeze(0).cpu()
            hr_4d = hr_cropped[i].unsqueeze(0).cpu()
            n = pl_module.eval_config.crop_border
            if n > 0:
                sr_4d = sr_4d[..., n:-n, n:-n]
                hr_4d = hr_4d[..., n:-n, n:-n]
            psnr_tensors = pl_module._build_psnr_tensors(sr_4d, hr_4d)
            psnr_dict = {
                key: torchmetrics.functional.image.peak_signal_noise_ratio(
                    *psnr_tensors[key], data_range=1.0
                ).item()
                for key in pl_module.eval_config.psnr_keys
            }
            ssim = torchmetrics.functional.image.structural_similarity_index_measure(
                sr_4d, hr_4d, data_range=1.0
            )
            self._buffer[dataset_name].append(
                (
                    filename,
                    lr_img[i].cpu() if should_log_images else None,
                    sr[i].cpu() if should_log_images else None,
                    hr_img[i].cpu() if should_log_images else None,
                    psnr_dict,
                    ssim.item(),
                )
            )

    def _flush_buffer(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        should_log_images: bool,
    ):
        """Log per-set means and (optionally) image strips for buffered batches.

        Shared between :meth:`on_validation_epoch_end` and
        :meth:`on_test_epoch_end`. Looks up the TensorBoard logger from
        ``trainer.loggers``; silently skips image emission for sets whose
        logger is missing.

        Args:
            trainer (lightning.Trainer): The active trainer (used for
                ``global_step`` and the TensorBoard logger lookup).
            pl_module (lightning.LightningModule): Receives ``log()`` calls
                for the per-set mean metrics.
            should_log_images (bool): When True, bicubic|SR|HR image strips
                are emitted to TensorBoard alongside the scalar means.
        """
        step = trainer.global_step

        for dataset_name, samples in self._buffer.items():
            if not samples:
                continue

            psnr_keys = samples[0][4].keys()
            mean_ssim = sum(s for *_, s in samples) / len(samples)

            for key in psnr_keys:
                mean_psnr = sum(s[4][key] for s in samples) / len(samples)
                pl_module.log(f"{dataset_name}_psnr({key})", mean_psnr, add_dataloader_idx=False)
            pl_module.log(f"{dataset_name}_ssim", mean_ssim, add_dataloader_idx=False)

            if should_log_images:
                tb_logger = next(
                    (
                        logger
                        for logger in trainer.loggers
                        if isinstance(logger, lightning.pytorch.loggers.TensorBoardLogger)
                    ),
                    None,
                )
                if tb_logger is None:
                    continue

                experiment = tb_logger.experiment
                for filename, lr, sr, hr, psnr_dict, ssim in samples:
                    for key, psnr_val in psnr_dict.items():
                        experiment.add_scalar(
                            f"{dataset_name}_psnr({key})/{filename}", psnr_val, global_step=step
                        )
                    experiment.add_scalar(f"{dataset_name}_ssim/{filename}", ssim, global_step=step)

                    # Triptych: bicubic | SR | HR, all at HR size.
                    # The bicubic panel is the LR upsampled to HR via bicubic
                    # interpolation — the standard SR baseline.  For SRCNN the
                    # LR is already at HR size (pre-upsampled in the dataset),
                    # so this resamples at 1:1 and is near-identity.  For
                    # SRResNet (LR < HR) it upscales to HR for direct visual
                    # comparison against SR.  SR is center-padded only when
                    # smaller than HR (SRCNN with padding='valid') to preserve
                    # full HR context on the right.
                    target_hw = hr.shape[-2:]
                    bicubic = self._bicubic_to(lr, target_hw)
                    sr_padded = self._pad_to_match(sr, target_hw)
                    strip = torchvision.utils.make_grid(
                        [bicubic, sr_padded, hr], nrow=3, padding=2, pad_value=0.5
                    )
                    experiment.add_image(f"{dataset_name}/{filename}", strip, global_step=step)

        self._buffer.clear()

    @staticmethod
    def _bicubic_to(img: torch.Tensor, target_hw: tuple) -> torch.Tensor:
        """Bicubic-upsample a ``(C, H, W)`` tensor to *target_hw*, clamped to [0, 1].

        Used to render the bicubic baseline panel of the triptych at HR size.
        Bicubic can overshoot for high-contrast inputs, so the output is
        clamped to the unit interval to keep the panel renderable as a
        normalized image.

        Args:
            img (torch.Tensor): Image tensor of shape ``(C, H, W)``.
            target_hw (tuple): Target ``(H, W)`` spatial dimensions.

        Returns:
            torch.Tensor: Bicubic-resampled tensor of shape ``(C, target_H, target_W)``.
        """
        out = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=target_hw, mode="bicubic", align_corners=False
        ).squeeze(0)
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _pad_to_match(img: torch.Tensor, target_hw: tuple) -> torch.Tensor:
        """Zero-pad a ``(C, H, W)`` tensor to *target_hw* spatial size.

        Args:
            img (torch.Tensor): Image tensor of shape ``(C, H, W)``.
            target_hw (tuple): Target ``(H, W)`` spatial dimensions.

        Returns:
            torch.Tensor: Padded tensor of shape ``(C, target_H, target_W)``.
        """
        target_h, target_w = target_hw
        _, h, w = img.shape

        pad_lr = (target_w - w) // 2
        pad_ud = (target_h - h) // 2
        img = torch.nn.functional.pad(img, (pad_lr, pad_lr, pad_ud, pad_ud), value=0.0)

        if img.shape[1] != target_h or img.shape[2] != target_w:
            # If target size is odd and img is even (or vice versa), add one more pixel of
            # padding to the right/bottom
            pad_right = target_w - img.shape[2]
            pad_bottom = target_h - img.shape[1]
            img = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom), value=0.0)

        return img


class GradNormLogger(Callback):
    """Log the total gradient L2 norm to TensorBoard periodically.

    Computes ``sqrt(sum(p.grad.norm() ** 2))`` across all model parameters
    after the backward pass and logs it as the ``"grad_norm"`` scalar.

    Args:
        log_every_n_steps (int): Compute and log every *n* training steps.
            Defaults to ``100``.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """Compute and log gradient norm if on the right step cadence.

        Args:
            trainer (lightning.Trainer): The trainer instance.
            pl_module (lightning.LightningModule): The model being trained.
        """
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().norm(2).item() ** 2
        total_norm = math.sqrt(total_norm_sq)

        pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False)


class WeightHistogramLogger(Callback):
    """Log the weights of the model as histograms to TensorBoard periodically,
    grouped by parameter prefixes (e.g., model.feat, model.mapping, model.recon).

    Args:
        log_every_n_steps (int): Log histograms every *n* training steps.
            Defaults to ``100``.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Log grouped weights as histograms if on the right step cadence.

        Args:
            trainer (lightning.Trainer): The trainer instance.
            pl_module (lightning.LightningModule): The model being trained.
            outputs (Any): The outputs of the model.
            batch (Any): The current batch.
            batch_idx (int): The index of the batch.
        """
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        tb_logger = next(
            (
                logger
                for logger in trainer.loggers
                if isinstance(logger, lightning.pytorch.loggers.TensorBoardLogger)
            ),
            None,
        )
        if tb_logger is None:
            return

        experiment = tb_logger.experiment

        for name, param in pl_module.named_parameters():
            if param.requires_grad and name.startswith("model."):
                parts = name.split(".", 2)
                tb_name = parts[0] + "." + "/".join(parts[1:])
                experiment.add_histogram(tb_name, param, global_step=trainer.global_step)


class SRCheckpoint(ModelCheckpoint):
    """Model checkpoint that monitors a super-resolution quality metric.

    A thin convenience wrapper around
    :class:`~lightning.pytorch.callbacks.ModelCheckpoint` that
    automatically sets ``mode='max'`` (both PSNR and SSIM are
    higher-is-better) and builds a descriptive filename pattern.

    The filename's metric *label* is sanitised (``/`` -> ``_``) and
    ``auto_insert_metric_name`` is disabled, so a ``/``-bearing
    ``monitor_metric`` (e.g. a future ``psnr/val/RGB`` TensorBoard-style tag)
    cannot leak a raw ``/`` into the filename template. Lightning's
    ``_format_checkpoint_name`` does no sanitisation itself, and with
    ``auto_insert_metric_name=True`` (the default) a raw ``/`` there causes
    ``TorchCheckpointIO`` to silently create a nested directory tree per
    save instead of a flat file. The ``metrics`` dict lookup that supplies
    the interpolated *value* is unaffected — it is a plain dict key and
    handles ``/`` fine.

    Args:
        monitor_metric (str): The validation metric to monitor.
            Defaults to ``"val_psnr(RGB)"``.  Use any ``val_psnr({key})``
            logged by the lightning module (e.g. ``"val_psnr(Y)"``,
            ``"val_psnr(YCbCr)"``) or ``"val_ssim"``.
        save_top_k (int): Number of best checkpoints to keep.
            Defaults to ``3``.
        dirpath (str | None): Directory to save checkpoints.
        filename_prefix (str): Prefix for checkpoint filenames.
            Defaults to ``"sr"``.
        **kwargs: Extra keyword arguments forwarded to
            :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    def __init__(
        self,
        monitor_metric: str = "val_psnr(RGB)",
        save_top_k: int = 3,
        dirpath: str | None = None,
        filename_prefix: str = "sr",
        mode: str = "max",
        **kwargs: Any,
    ):
        label = monitor_metric.replace("/", "_")
        filename = f"{filename_prefix}-{{step}}-{label}={{{monitor_metric}:.4f}}"
        super().__init__(
            monitor=monitor_metric,
            mode=mode,
            save_top_k=save_top_k,
            dirpath=dirpath,
            filename=filename,
            auto_insert_metric_name=False,
            **kwargs,
        )

    def setup(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage: str,
    ) -> None:
        """Validate ``monitor`` against the metrics ``SRLightning`` will log.

        Lightning itself only raises ``MisconfigurationException`` for a
        missing monitor once the val loop has already run at least once
        (``ModelCheckpoint._save_topk_checkpoint``, gated on
        ``val_loop._has_run``) — with a long ``val_check_interval`` that is a
        late, expensive-to-reach crash. This moves the same failure to
        startup by checking ``monitor`` against ``pl_module.eval_config.psnr_keys``
        (the same seam ``SRLightning`` logs val PSNR from) up front, before
        any training happens.

        Args:
            trainer (lightning.Trainer): The active trainer.
            pl_module (lightning.LightningModule): The model being trained;
                must expose ``eval_config`` (an :class:`SREvalConfig`).
            stage (str): Lightning trainer stage.

        Raises:
            MisconfigurationException: If ``monitor`` does not name a
                metric ``SRLightning`` will log for ``stage``.
        """
        super().setup(trainer, pl_module, stage)
        valid_metrics = {f"val_psnr({key})" for key in pl_module.eval_config.psnr_keys}
        valid_metrics.add("val_ssim")
        if self.monitor is not None and self.monitor not in valid_metrics:
            raise MisconfigurationException(
                f"`SRCheckpoint(monitor_metric={self.monitor!r})` does not match any "
                f"metric `SRLightning` will log: {sorted(valid_metrics)}. HINT: check "
                f"`eval_config.psnr_channels` / `eval_config.separate_psnr`, or monitor "
                f"`val_ssim`."
            )
