"""Lightning callbacks for SR training.

:class:`BenchmarkImageLogger` logs per-image bicubic|SR|HR strips and PSNR/SSIM
for held-out test sets (Set5, Set14, ...) during both ``cli fit`` (every
N val cycles) and ``cli test`` (one-shot final eval).
:class:`GradNormLogger` and :class:`WeightHistogramLogger` log diagnostic
training signals; :class:`SRCheckpoint` is a thin
:class:`~lightning.pytorch.callbacks.ModelCheckpoint` preset for SR metrics;
:class:`SRPredictionWriter` writes ``cli predict`` output to disk as PNGs.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning
import torch
import torch.nn.functional
import torchmetrics.functional
import torchvision
from lightning.pytorch.callbacks import BasePredictionWriter, Callback, ModelCheckpoint
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class BenchmarkImageLogger(Callback):
    """Logs bicubic|SR|HR image composites and PSNR/SSIM to TensorBoard for held-out sets.

    Covers one or more held-out test/benchmark dataloaders (Set5, Set14, …).
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

    Per-set means go under ``"psnr/{name}/{key}"`` and ``"ssim/{name}/{key}"``
    every cycle. Per-image scalars under ``"per_image/{name}/psnr/{key}/{filename}"``
    and ``"per_image/{name}/ssim/{key}/{filename}"`` are gated by
    ``log_per_image_metrics`` (default off — see that arg).

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
        log_per_image_metrics: Emit the ``per_image/...`` PSNR/SSIM scalars
            (one series per image per key). Default ``False``: with N test
            images and both PSNR/SSIM keyed by colorspace, this is
            ``N * (psnr_keys + ssim_keys)`` TensorBoard tags — e.g. 76 series
            for 19 images at template defaults, 1190 with BSD100 plus
            ``separate_psnr``. The cost is tag-count (TensorBoard's scalar
            pane stops being navigable), not bytes — these scalars are only
            ~285 KB. Per-set means are unaffected and always logged.
    """

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        log_every_n_val_runs: int = 5,
        log_per_image_metrics: bool = False,
    ):
        super().__init__()
        self.dataset_names = list(dataset_names) if dataset_names else None
        self.log_every_n_val_runs = log_every_n_val_runs
        self.log_per_image_metrics = log_per_image_metrics

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
            trainer: The active trainer.
            pl_module: Unused.
            stage: Unused — both val and test mappings are built up-front.
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
            trainer: Unused.
            pl_module: Unused.
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
            trainer: The active trainer.
            pl_module: The model being evaluated.
            outputs: Unused — :meth:`SRLightning.validation_step` returns
                ``None`` for idx >= 1.
            batch: ``(lr_img, hr_img)`` tuple from the test loader.
            batch_idx: Index of the batch within the current dataloader.
            dataloader_idx: Index of the loader within
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
            trainer: The active trainer.
            pl_module: The model being evaluated.
        """
        self._flush_buffer(
            trainer=trainer,
            pl_module=pl_module,
            should_log_images=self._on_image_log_interval(),
        )

    def on_test_epoch_start(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """Clear buffers ahead of a test run.

        Args:
            trainer: Unused.
            pl_module: Unused.
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
            trainer: The active trainer.
            pl_module: The model being evaluated.
            outputs: Unused — :meth:`SRLightning.test_step` returns ``None``.
            batch: ``(lr_img, hr_img)`` tuple from the test loader.
            batch_idx: Index of the batch within the current dataloader.
            dataloader_idx: Index of the loader within
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
            trainer: The active trainer.
            pl_module: The model being evaluated.
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
        *source_dataloaders* recovers the underlying dataset for filename
        resolution. When *should_log_images*, LR/SR/HR tensors are cached for
        image-strip emission in :meth:`_flush_buffer`; otherwise only scalar
        metrics are buffered.
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
            # Still a private reach (P5.9): the colorspace split has no public
            # seam, and duplicating it here would recreate the divergence P2.1
            # removed. Keys now come from eval_config, so this is a value
            # lookup only — the callback no longer decides *which* keys exist.
            metric_tensors = pl_module._build_metric_tensors(sr_4d, hr_4d)
            psnr_dict = {
                key: torchmetrics.functional.image.peak_signal_noise_ratio(
                    *metric_tensors[key], data_range=1.0
                ).item()
                for key in pl_module.eval_config.psnr_keys
            }
            ssim_dict = {
                key: torchmetrics.functional.image.structural_similarity_index_measure(
                    *metric_tensors[key], data_range=1.0
                ).item()
                for key in pl_module.eval_config.ssim_keys
            }
            self._buffer[dataset_name].append(
                (
                    filename,
                    lr_img[i].cpu() if should_log_images else None,
                    sr[i].cpu() if should_log_images else None,
                    hr_img[i].cpu() if should_log_images else None,
                    psnr_dict,
                    ssim_dict,
                )
            )

    def _flush_buffer(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        should_log_images: bool,
    ):
        """Log per-set means and (optionally) per-image scalars/image strips.

        Shared between :meth:`on_validation_epoch_end` and
        :meth:`on_test_epoch_end`. Looks up the TensorBoard logger from
        ``trainer.loggers``; silently skips per-image/image emission for
        sets whose logger is missing. Per-image scalar emission is gated by
        ``self.log_per_image_metrics``, independent of *should_log_images*
        (which gates only the image strips) — the two concerns cost
        differently (tag count vs. bytes) and are configured separately.
        """
        step = trainer.global_step

        for dataset_name, samples in self._buffer.items():
            if not samples:
                continue

            psnr_keys = samples[0][4].keys()
            ssim_keys = samples[0][5].keys()

            for key in psnr_keys:
                mean_psnr = sum(s[4][key] for s in samples) / len(samples)
                pl_module.log(f"psnr/{dataset_name}/{key}", mean_psnr, add_dataloader_idx=False)
            for key in ssim_keys:
                mean_ssim = sum(s[5][key] for s in samples) / len(samples)
                pl_module.log(f"ssim/{dataset_name}/{key}", mean_ssim, add_dataloader_idx=False)

            if should_log_images or self.log_per_image_metrics:
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
                for filename, lr, sr, hr, psnr_dict, ssim_dict in samples:
                    if self.log_per_image_metrics:
                        for key, psnr_val in psnr_dict.items():
                            experiment.add_scalar(
                                f"per_image/{dataset_name}/psnr/{key}/{filename}",
                                psnr_val,
                                global_step=step,
                            )
                        for key, ssim_val in ssim_dict.items():
                            experiment.add_scalar(
                                f"per_image/{dataset_name}/ssim/{key}/{filename}",
                                ssim_val,
                                global_step=step,
                            )

                    if not should_log_images:
                        continue

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
        """
        out = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=target_hw, mode="bicubic", align_corners=False
        ).squeeze(0)
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _pad_to_match(img: torch.Tensor, target_hw: tuple) -> torch.Tensor:
        """Zero-pad a ``(C, H, W)`` tensor to *target_hw* spatial size."""
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
    after the backward pass and logs it as the ``"diag/grad_norm"`` scalar.

    Args:
        log_every_n_steps (int): Compute and log every *n* training steps.
            Defaults to ``100``.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """Compute and log gradient norm if on the right step cadence.

        Per-parameter norms are stacked and reduced with a single
        ``torch.linalg.vector_norm`` call so only one ``.item()`` forces a
        GPU sync, instead of one per parameter (the previous per-parameter
        ``.item()`` accumulation loop).

        Args:
            trainer: The trainer instance.
            pl_module: The model being trained.
        """
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        grad_norms = [p.grad.detach().norm(2) for p in pl_module.parameters() if p.grad is not None]
        total_norm = (
            torch.linalg.vector_norm(torch.stack(grad_norms), ord=2).item() if grad_norms else 0.0
        )

        pl_module.log("diag/grad_norm", total_norm, on_step=True, on_epoch=False)


class WeightHistogramLogger(Callback):
    """Logs model weights as TensorBoard histograms, grouped by parameter prefix.

    Groups by prefix, e.g. ``model.feat``, ``model.mapping``, ``model.recon``.

    Args:
        log_every_n_steps (int): Log histograms every *n* training steps.
            Defaults to ``10000``. Histograms dominate event-file size, and
            one is written per tracked parameter on every cadence hit — at
            the templates' 1M-step schedule the old default of ``100`` was
            ~10k writes per parameter (multi-GB event files) despite never
            having been exercised at that scale (neither tracked template
            wires this callback). ``10000`` drops that to ~100
            writes per parameter, still enough to see the weight-drift
            trend across a full run.
    """

    def __init__(self, log_every_n_steps: int = 10000):
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
            trainer: The trainer instance.
            pl_module: The model being trained.
            outputs: Unused.
            batch: Unused.
            batch_idx: Unused.
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
    ``monitor_metric`` (e.g. the default ``psnr/val/RGB`` TensorBoard-hierarchy
    tag) cannot leak a raw ``/`` into the filename template. Lightning's
    ``_format_checkpoint_name`` does no sanitisation itself, and with
    ``auto_insert_metric_name=True`` (the default) a raw ``/`` there causes
    ``TorchCheckpointIO`` to silently create a nested directory tree per
    save instead of a flat file. The ``metrics`` dict lookup that supplies
    the interpolated *value* is unaffected — it is a plain dict key and
    handles ``/`` fine.

    Args:
        monitor_metric: The validation metric to monitor. Any ``psnr/val/{key}``
            or ``ssim/val/{key}`` logged by the lightning module (e.g.
            ``"psnr/val/Y"``, ``"psnr/val/YCbCr"``, ``"ssim/val/RGB"``).
        save_top_k: Number of best checkpoints to keep.
        dirpath: Directory to save checkpoints.
        filename_prefix: Prefix for checkpoint filenames.
        **kwargs: Extra keyword arguments forwarded to
            :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    def __init__(
        self,
        monitor_metric: str = "psnr/val/RGB",
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
        / ``ssim_keys`` (the same seams ``SRLightning`` logs val PSNR/SSIM
        from) up front, before any training happens.

        Args:
            trainer: The active trainer.
            pl_module: The model being trained; must expose ``eval_config``
                (an :class:`SREvalConfig`).
            stage: Lightning trainer stage. Only ``"fit"`` is checked — the
                monitored tags are val-loop metrics, so they are never logged
                under ``validate``/``test``/``predict`` and demanding them
                there would reject configs that are perfectly valid for the
                stage actually being run.

        Raises:
            MisconfigurationException: If ``monitor`` does not name a
                metric ``SRLightning`` will log during ``fit``.
        """
        super().setup(trainer, pl_module, stage)
        if stage != "fit":
            return
        valid_metrics = {f"psnr/val/{key}" for key in pl_module.eval_config.psnr_keys}
        valid_metrics |= {f"ssim/val/{key}" for key in pl_module.eval_config.ssim_keys}
        if self.monitor is not None and self.monitor not in valid_metrics:
            raise MisconfigurationException(
                f"`SRCheckpoint(monitor_metric={self.monitor!r})` does not match any "
                f"metric `SRLightning` will log: {sorted(valid_metrics)}. HINT: check "
                f"`eval_config.psnr_channels` / `eval_config.separate_psnr`, or "
                f"`eval_config.ssim_channels`."
            )


class SRPredictionWriter(BasePredictionWriter):
    """Writes ``predict_step`` output as 8-bit PNGs named after their input files.

    ``cli predict`` has no HR reference to score against, so a saved image
    *is* the entire result — this callback is what makes the subcommand
    produce a visible artifact instead of a discarded in-memory tensor.

    Only ``write_interval='batch'`` is supported — ``'epoch'`` /
    ``'batch_and_epoch'`` would accumulate every prediction in memory for
    the whole run before writing, which defeats writing PNGs one-by-one in
    the first place, and this class doesn't implement
    ``write_on_epoch_end``.

    Args:
        output_dir: Directory PNGs are written to; created (including
            parents) if missing.
    """

    def __init__(self, output_dir: str | Path):
        super().__init__("batch")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        prediction: torch.Tensor,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Save each SR image in ``prediction`` as ``<input_stem>.png``.

        Filenames are resolved from the predict dataloader's underlying
        dataset via ``batch_indices`` and its ``.img_paths`` contract — the
        same one :class:`BenchmarkImageLogger` relies on — rather than from
        ``batch`` itself, which is a bare LR tensor with no filename.

        Args:
            trainer: The active trainer.
            pl_module: Unused.
            prediction: ``SRLightning.predict_step`` output — SR RGB batch,
                ``float32`` in ``[0, 1]``, shape ``(B, 3, H, W)``.
            batch_indices: Dataset indices of the samples in this batch,
                supplied by Lightning's predict loop.
            batch: Unused — filenames come from ``batch_indices``, not the
                tensor itself.
            batch_idx: Unused.
            dataloader_idx: Index of the predict loader — used only when
                ``trainer.predict_dataloaders`` is a list.
        """
        loader = trainer.predict_dataloaders
        if isinstance(loader, list | tuple):
            loader = loader[dataloader_idx]
        img_paths = loader.dataset.img_paths

        for i, sample_idx in enumerate(batch_indices):
            filename = img_paths[sample_idx].stem
            out_path = self.output_dir / f"{filename}.png"
            torchvision.utils.save_image(prediction[i].clamp(0.0, 1.0), out_path)
