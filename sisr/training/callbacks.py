"""Lightning callbacks for SR training.

:class:`BenchmarkImageLogger` logs per-image bicubic|SR|HR strips and PSNR/SSIM
for held-out test sets (Set5, Set14, ...) during both ``cli fit`` (every
N val cycles) and ``cli test`` (one-shot final eval).
:class:`GradNormLogger` and :class:`WeightHistogramLogger` log diagnostic
training signals; :class:`SRCheckpoint` is a thin
:class:`~lightning.pytorch.callbacks.ModelCheckpoint` preset for SR metrics;
:class:`SRWeightsCheckpoint` is a sibling preset that saves bare, optimizer-free
safetensors weights instead; :class:`SRPredictionWriter` writes ``cli predict``
output to disk as PNGs.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast
from weakref import proxy

import lightning
import torch
import torch.nn.functional
import torchvision
from lightning.pytorch.callbacks import BasePredictionWriter, Callback, ModelCheckpoint
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from .. import artifacts
from ..metrics.perceptual import PERCEPTUAL_METRICS
from .metadata import build_component_metadata, build_metadata

if TYPE_CHECKING:  # `SRLightning` is referenced in annotations only, and a
    # runtime import would be circular -- it imports this module's package.
    from .lightning_module import SRLightning


def _sr(pl_module: lightning.LightningModule) -> "SRLightning":
    """Narrow Lightning's hook parameter to the module these callbacks require.

    Every hook here is typed by Lightning as taking a bare ``LightningModule``,
    while everything they read -- ``eval_config``, ``scorer``, ``predict_rgb``,
    the provenance builders -- belongs to :class:`SRLightning`. Overriding a hook
    with a narrower parameter type is unsound and rejected, so the narrowing
    happens at the boundary instead, once, with a name.

    A cast rather than an ``isinstance`` check on purpose: a module without those
    attributes already fails at exactly the same call today, and turning that into
    a different failure is a behaviour change this seam has no business making.

    Args:
        pl_module: The module Lightning handed the hook.

    Returns:
        The same object, typed as what it has to be.
    """
    return cast("SRLightning", pl_module)


class BenchmarkSample(NamedTuple):
    """One buffered per-image result: PSNR/SSIM only, no image tensors.

    Fields are accessed by name at every read site — a bare positional
    tuple here previously made ``_flush_buffer``'s mean computation
    (``s[4]``/``s[5]``) unreadable at the call site.

    Attributes:
        filename: Stem of the source image path, used as the TensorBoard tag suffix.
        psnr: PSNR value per configured key (``eval_config.psnr_keys``).
        ssim: SSIM value per configured key (``eval_config.ssim_keys``).
        perceptual: Perceptual score per configured metric (``eval_config.perceptual_keys``).
    """

    filename: str
    psnr: dict[str, float]
    ssim: dict[str, float]
    perceptual: dict[str, float]


def _logger_step(trainer: lightning.Trainer) -> int:
    """Return the step axis Lightning writes ``self.log`` metrics on.

    Deliberately not ``trainer.global_step``, which counts *optimizer* steps: a
    module under manual optimization with two optimizers (SRGAN steps D and G
    per batch) advances it twice per batch, so anything logged against it lands
    at 2x the step of every ``self.log`` scalar. Lightning maintains
    ``_batches_that_stepped`` for precisely this reason — "increased once per
    batch disregarding multiple optimizers on purpose for loggers" — and reads
    it back in ``logger_connector``; its own ``LearningRateMonitor`` and
    ``DeviceStatsMonitor`` reach for the same private attribute, which is what
    makes matching them here the correct call rather than a shortcut.

    Args:
        trainer: The active trainer.

    Returns:
        The batch-counted step shared by every ``self.log`` metric.
    """
    return trainer.fit_loop.epoch_loop._batches_that_stepped


class BenchmarkImageLogger(Callback):
    """Logs bicubic|SR|HR image composites and PSNR/SSIM to TensorBoard for held-out sets.

    Covers one or more held-out test/benchmark dataloaders (Set5, Set14, …).
    Fires during *both* validation and test stages:

    * **During `cli fit` / `cli validate`** — `SRDataModule.val_dataloader()`
      returns ``[primary_val] + [test_loaders...]``; this callback ignores
      ``dataloader_idx == 0`` (handled by `SRLightning.validation_step`) and
      processes indices ``1..N`` against ``dataset_names``.  Images are
      logged every *n*-th val run (controlled by ``every_n_val_runs``)
      so TensorBoard storage stays bounded over long training schedules.
    * **During `cli test`** — `SRDataModule.test_dataloader()` returns the
      test loaders only (no primary), so indices ``0..N-1`` map straight to
      ``dataset_names``.  Images are logged every test run (no throttle —
      ``cli test`` is typically a one-shot final-eval invocation).

    Each image lands as a horizontal **bicubic | SR | HR** strip at HR scale:
    the LR is bicubic-upsampled to HR size (the standard SR baseline), and the
    SR output is center-padded when smaller than HR (``padding='valid'``) so
    all three panels share the same spatial size.

    Per-set means go under ``"psnr/{name}/{key}"``, ``"ssim/{name}/{key}"`` and,
    for each metric in ``eval_config.perceptual_keys``, ``"{metric}/{name}"``
    (e.g. ``"lpips/Set5"``) every cycle. Per-image scalars under
    ``"per_image/{name}/psnr/{key}/{filename}"`` and
    ``"per_image/{name}/ssim/{key}/{filename}"`` are gated by
    ``log_per_image_metrics`` (default off — see that arg).

    Border cropping is sourced from ``pl_module.eval_config.crop_border`` at
    the use site; this avoids dual-knob configuration drift.

    Metrics are computed directly on the on-device SR/HR slices (mirroring
    the primary-val-loader path in ``SRLightning.validation_step``) and both
    the per-image scalars and the image strip are emitted immediately, per
    image, from :meth:`_collect_batch` — nothing image-shaped is buffered.
    Buffering full-resolution LR/SR/HR CPU tensors for every image across
    every test set until epoch end cost ~0.5 GB per validation cycle purely
    to defer ``add_image``; only a :class:`BenchmarkSample` (filename +
    PSNR/SSIM/perceptual dicts) is buffered now, for the per-set mean computed in
    :meth:`_flush_buffer`. The TensorBoard experiment is resolved once, in
    :meth:`setup`, rather than re-searched on every batch/epoch-end.

    Args:
        dataset_names: Ordered list of test set names (e.g.
            ``["Set5", "Set14"]``) matching the order of
            ``SRDataModule.test_dataloader()`` / the trailing entries of
            ``val_dataloader()``.  When ``None`` the callback auto-discovers
            from ``trainer.datamodule.test_names``.
        every_n_val_runs: Image-strip throttle for the val stage.  With
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
        every_n_val_runs: int = 5,
        log_per_image_metrics: bool = False,
    ):
        super().__init__()
        # [] rather than None for 'not given': an empty list already reached the
        # same auto-discovery branch in setup(), so the second sentinel bought
        # nothing but an Optional every reader had to re-narrow.
        self.dataset_names: list[str] = list(dataset_names) if dataset_names else []
        self.every_n_val_runs = every_n_val_runs
        self.log_per_image_metrics = log_per_image_metrics

        # Val: primary val is at idx 0, test sets at 1..N → {1: 'Set5', ...}
        # Test: only test sets present, at idx 0..N-1     → {0: 'Set5', ...}
        self._val_mapping: dict[int, str] = {}
        self._test_mapping: dict[int, str] = {}

        self._val_run_count = 0

        self._buffer: dict[str, list[BenchmarkSample]] = {}

        # Resolved once in setup() rather than re-searched every batch/epoch-end;
        # None when no TensorBoard logger is attached (image/scalar emission is
        # then silently skipped, same as the previous per-flush behaviour).
        self._tb_experiment: Any = None

    def setup(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage: str,
    ) -> None:
        """Resolve ``dataset_names``, both dataloader_idx mappings, and the TB experiment.

        Auto-discovers ``dataset_names`` from ``trainer.datamodule.test_names``
        when not supplied at construction time.

        Args:
            trainer: The active trainer.
            pl_module: Unused.
            stage: Unused — both val and test mappings are built up-front.
        """
        if not self.dataset_names:
            dm = getattr(trainer, "datamodule", None)
            names = getattr(dm, "test_names", None) if dm is not None else None
            self.dataset_names = list(names or [])
        self._val_mapping = {i + 1: name for i, name in enumerate(self.dataset_names)}
        self._test_mapping = {i: name for i, name in enumerate(self.dataset_names)}

        tb_logger = next(
            (
                logger
                for logger in getattr(trainer, "loggers", None) or []
                if isinstance(logger, lightning.pytorch.loggers.TensorBoardLogger)
            ),
            None,
        )
        self._tb_experiment = tb_logger.experiment if tb_logger is not None else None

    def on_validation_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
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
    ) -> None:
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
    ) -> None:
        """Log per-set mean PSNR/SSIM for the val epoch.

        Image strips and per-image scalars were already streamed to
        TensorBoard per-image in :meth:`_collect_batch`; only the means
        remain to compute here.

        Args:
            trainer: Unused.
            pl_module: The model being evaluated.
        """
        self._flush_buffer(pl_module=pl_module)

    def on_test_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
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
    ) -> None:
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

    def on_test_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Log per-set mean PSNR/SSIM at the end of ``cli test``.

        Image strips and per-image scalars were already streamed to
        TensorBoard per-image in :meth:`_collect_batch`; only the means
        remain to compute here.

        Args:
            trainer: Unused.
            pl_module: The model being evaluated.
        """
        self._flush_buffer(pl_module=pl_module)

    def _on_image_log_interval(self) -> bool:
        """Return True when this val run should also log image strips."""
        return self._val_run_count % self.every_n_val_runs == 0

    def _collect_batch(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        batch: Any,
        batch_idx: int,
        dataset_name: str,
        source_dataloaders: Sequence[Any] | None,
        dataloader_idx: int,
        should_log_images: bool,
    ) -> None:
        """Forward one batch, compute per-image metrics on-device, and stream image strips.

        Shared between :meth:`on_validation_batch_end` and
        :meth:`on_test_batch_end`. Border cropping is sourced from
        ``pl_module.eval_config.crop_border`` at the use site.
        *source_dataloaders* recovers the underlying dataset for filename
        resolution.

        PSNR/SSIM/perceptual are computed from the on-device ``sr``/``hr_cropped``
        slices — mirroring the primary-val-loader path in
        ``SRLightning.validation_step`` — so only the scalar ``.item()``
        results ever leave the GPU. SSIM and perceptual scores go through
        ``pl_module.scorer`` — the same object ``validation_step`` uses — rather
        than calling ``torchmetrics``/``perceptual_score`` here directly, so
        this callback cannot silently diverge from ``validation_step`` on the
        crop, the reductions, or which SSIM/perceptual backbone a given
        ``eval_config`` means. When
        *should_log_images* (or
        ``self.log_per_image_metrics``), exactly one host transfer per image
        (``lr_img[i]``/``sr[i]``/``hr_img[i]`` each ``.cpu()`` at most once)
        composes the bicubic|SR|HR strip and/or the per-image scalars, and
        both are emitted to TensorBoard immediately — nothing image-shaped is
        buffered. Only ``BenchmarkSample(filename, psnr_dict, ssim_dict,
        perceptual_dict)`` is appended to ``self._buffer``, for
        :meth:`_flush_buffer`'s per-set mean.
        """
        lr_img, hr_img = batch

        with torch.no_grad():
            sr, hr_cropped = _sr(pl_module).predict_rgb(lr_img, hr_img)

        # Resolve filenames from the underlying dataset for use as TB tags.
        # Optional only because Lightning declares the loader lists for every
        # stage; the hooks that call this one fire only while theirs is set.
        assert source_dataloaders is not None, "no dataloaders for the running stage"
        dataset = source_dataloaders[dataloader_idx].dataset
        batch_size = lr_img.size(0)
        # Resolved once per batch, and only when something will actually be
        # emitted — a trainer with no fit loop is legitimate when no TB logger
        # is attached, and the guard below already skips every emission then.
        step = _logger_step(trainer) if self._tb_experiment is not None else 0

        for i in range(batch_size):
            global_idx = batch_idx * batch_size + i
            filename = dataset.img_paths[global_idx].stem

            # One scorer object, shared with validation_step, so the two paths
            # cannot disagree on the crop, the colorspace split, the PSNR
            # reduction or which SSIM eval_config.ssim_impl names. This used to
            # be four separate reaches into private SRLightning methods, each
            # of which could have been "tidied" independently — PSNR in fact
            # was, calling torchmetrics with the default dim=None (whole-batch
            # pooling) and agreeing with validation_step only because this loop
            # slices one image at a time.
            scores = _sr(pl_module).scorer.score(sr[i : i + 1], hr_cropped[i : i + 1])
            psnr_dict = {key: value.item() for key, value in scores.psnr.items()}
            ssim_dict = {key: value.item() for key, value in scores.ssim.items()}
            perceptual_dict = {name: value.item() for name, value in scores.perceptual.items()}
            self._buffer[dataset_name].append(
                BenchmarkSample(filename, psnr_dict, ssim_dict, perceptual_dict)
            )

            if self._tb_experiment is None:
                continue

            if self.log_per_image_metrics:
                for key, psnr_val in psnr_dict.items():
                    tag = f"per_image/{dataset_name}/psnr/{key}/{filename}"
                    self._tb_experiment.add_scalar(tag, psnr_val, global_step=step)
                for key, ssim_val in ssim_dict.items():
                    tag = f"per_image/{dataset_name}/ssim/{key}/{filename}"
                    self._tb_experiment.add_scalar(tag, ssim_val, global_step=step)

            if not should_log_images:
                continue

            # One host transfer per image, only on cycles that log strips.
            lr_cpu = lr_img[i].cpu()
            sr_cpu = sr[i].cpu()
            hr_cpu = hr_img[i].cpu()

            # Triptych: bicubic | SR | HR, all at HR size.
            # The bicubic panel is the LR upsampled to HR via bicubic
            # interpolation — the standard SR baseline.  For SRCNN the
            # LR is already at HR size (pre-upsampled in the dataset),
            # so this resamples at 1:1 and is near-identity.  For
            # SRResNet (LR < HR) it upscales to HR for direct visual
            # comparison against SR.  SR is center-padded only when
            # smaller than HR (SRCNN with padding='valid') to preserve
            # full HR context on the right.
            target_hw = hr_cpu.shape[-2:]
            bicubic = self._bicubic_to(lr_cpu, target_hw)
            sr_padded = self._pad_to_match(sr_cpu, target_hw)
            strip = torchvision.utils.make_grid(
                [bicubic, sr_padded, hr_cpu], nrow=3, padding=2, pad_value=0.5
            )
            self._tb_experiment.add_image(f"{dataset_name}/{filename}", strip, global_step=step)

    def _flush_buffer(self, pl_module: lightning.LightningModule) -> None:
        """Log per-set mean PSNR/SSIM/perceptual scores from the buffered samples.

        Shared between :meth:`on_validation_epoch_end` and
        :meth:`on_test_epoch_end`. Per-image scalars and image strips were
        already streamed to TensorBoard from :meth:`_collect_batch`; this
        only reduces the buffered ``BenchmarkSample.psnr``/``.ssim``/``.perceptual``
        dicts to a mean per dataset/key. Perceptual tags are metric-first
        (``lpips/{dataset_name}``, ``dists/{dataset_name}``) with no ``perceptual/``
        prefix segment, matching ``psnr/{dataset_name}/{key}``'s hierarchy.
        """
        for dataset_name, samples in self._buffer.items():
            if not samples:
                continue

            psnr_keys = samples[0].psnr.keys()
            ssim_keys = samples[0].ssim.keys()

            for key in psnr_keys:
                mean_psnr = sum(s.psnr[key] for s in samples) / len(samples)
                pl_module.log(
                    f"psnr/{dataset_name}/{key}",
                    mean_psnr,
                    add_dataloader_idx=False,
                    sync_dist=True,
                )
            for key in ssim_keys:
                mean_ssim = sum(s.ssim[key] for s in samples) / len(samples)
                pl_module.log(
                    f"ssim/{dataset_name}/{key}",
                    mean_ssim,
                    add_dataloader_idx=False,
                    sync_dist=True,
                )
            for name in samples[0].perceptual:
                mean = sum(s.perceptual[name] for s in samples) / len(samples)
                pl_module.log(
                    f"{name}/{dataset_name}", mean, add_dataloader_idx=False, sync_dist=True
                )

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
        every_n_batches (int): Compute and log every *n* batches. Defaults to
            ``100``. The unit is in the name deliberately: Lightning's own
            callbacks do the same (``every_n_train_steps``, ``every_n_epochs``),
            and the old name was a homonym of ``Trainer.log_every_n_steps``,
            which means something else — the metric *flush* cadence. Batches
            also make the setting mean one thing under both automatic and manual
            optimization, and match the axis the emitted scalar is plotted on.
            See :func:`_logger_step`.
    """

    def __init__(self, every_n_batches: int = 100):
        super().__init__()
        self.every_n_batches = every_n_batches

    def on_after_backward(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Compute and log gradient norm if on the right step cadence.

        Per-parameter norms are stacked and reduced with a single
        ``torch.linalg.vector_norm`` call so only one ``.item()`` forces a
        GPU sync, instead of one per parameter (the previous per-parameter
        ``.item()`` accumulation loop).

        Args:
            trainer: The trainer instance.
            pl_module: The model being trained.
        """
        if _logger_step(trainer) % self.every_n_batches != 0:
            return

        grad_norms = [p.grad.detach().norm(2) for p in pl_module.parameters() if p.grad is not None]
        total_norm = (
            torch.linalg.vector_norm(torch.stack(grad_norms), ord=2).item() if grad_norms else 0.0
        )

        pl_module.log("diag/grad_norm", total_norm, on_step=True, on_epoch=False, sync_dist=True)


class WeightHistogramLogger(Callback):
    """Logs model weights as TensorBoard histograms, grouped by parameter prefix.

    Groups by prefix, e.g. ``model.feat``, ``model.mapping``, ``model.recon``.

    Args:
        every_n_batches (int): Log histograms every *n* batches, matching the
            axis they are written on — see :func:`_logger_step`. Named for its
            unit, following Lightning's own ``every_n_*`` callback arguments.
            Defaults to ``10000``. Histograms dominate event-file size, and
            one is written per tracked parameter on every cadence hit — at
            the templates' 1M-step schedule the old default of ``100`` was
            ~10k writes per parameter (multi-GB event files) despite never
            having been exercised at that scale (neither tracked template
            wires this callback). ``10000`` drops that to ~100
            writes per parameter, still enough to see the weight-drift
            trend across a full run.
    """

    def __init__(self, every_n_batches: int = 10000):
        super().__init__()
        self.every_n_batches = every_n_batches

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log grouped weights as histograms if on the right step cadence.

        Args:
            trainer: The trainer instance.
            pl_module: The model being trained.
            outputs: Unused.
            batch: Unused.
            batch_idx: Unused.
        """
        step = _logger_step(trainer)
        if step % self.every_n_batches != 0:
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
                experiment.add_histogram(tb_name, param, global_step=step)


if TYPE_CHECKING:
    # The mixin reads `monitor`, `dirpath`, `FILE_EXTENSION` and `_remove_checkpoint`
    # off the ModelCheckpoint it is combined with, and chains `on_train_start` through
    # it. Naming that base for the checker alone keeps the runtime MRO exactly as the
    # docstring below describes -- a mixin over `object`, so `super()` still lands on
    # ModelCheckpoint rather than ahead of it.
    _RollingBase = ModelCheckpoint
else:
    _RollingBase = object


class _RollingSaveMixin(_RollingBase):
    """Oldest-first deletion for the no-monitor (rolling) case.

    Lightning refuses ``ModelCheckpoint(monitor=None, save_top_k=N>1)``
    ("No quantity for top_k to track"), so a rolling window cannot be
    expressed through ``save_top_k`` at all. The legal construction is
    ``monitor=None, save_top_k=-1`` (keep everything) with the window
    enforced here. Verified on Lightning 2.6.5.

    Shared by :class:`SRCheckpoint` and :class:`SRWeightsCheckpoint` —
    siblings under :class:`~lightning.pytorch.callbacks.ModelCheckpoint`, not
    parent/child, so this logic can't be reused via one ``super()`` chain
    across both. The window bookkeeping itself lives in
    :meth:`_enforce_rolling_window`, a seam distinct from ``_save_checkpoint``:
    :class:`SRWeightsCheckpoint` already overrides ``_save_checkpoint`` to
    write a bare ``.safetensors`` payload instead of Lightning's full checkpoint, and
    that override does not call ``super()._save_checkpoint()`` — so a version
    of this mixin that put the bookkeeping inside its own ``_save_checkpoint``
    would be shadowed by that override and silently never run. Each
    subclass's ``_save_checkpoint`` calls :meth:`_enforce_rolling_window`
    explicitly instead.

    The window itself is never persisted in ``state_dict`` — it is rebuilt from
    the files already on disk when a run resumes (:meth:`on_train_start`), which
    is what keeps ``fit --ckpt_path`` from orphaning the pre-resume saves.
    """

    def _init_rolling(self, keep_last: int, filename_prefix: str) -> None:
        if keep_last < 1:
            raise ValueError(f"keep_last must be >= 1; got {keep_last}")
        self.keep_last = keep_last
        self._rolling: list[str] = []
        # Rolling filenames are exactly f"{filename_prefix}-{step}{FILE_EXTENSION}",
        # so this matches every file THIS callback writes and nothing else: a
        # sibling callback in the same dirpath differs in prefix or extension, and
        # `last.ckpt`/a monitored callback's metric-bearing name never match.
        self._rolling_pattern = re.compile(
            rf"{re.escape(filename_prefix)}_s(\d+){re.escape(self.FILE_EXTENSION)}"
        )

    def on_train_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Seed the rolling window from disk when this run is resuming.

        Nothing persists the window: ``ModelCheckpoint.state_dict`` carries
        ``monitor``/``best_*``/``kth_*``/``dirpath``/``last_model_path`` only. So
        without this, a ``fit --ckpt_path`` resume starts with an empty window and
        every pre-resume file is orphaned — never counted, never deleted, up to
        ``keep_last`` per callback per resume (three rolling callbacks run in the
        shipped adversarial template).

        Only a resume seeds. A fresh run into a populated ``dirpath`` leaves those
        files alone rather than adopting — and eventually deleting — checkpoints it
        did not write.

        Runs here rather than in ``setup`` because ``trainer.ckpt_path`` is not
        assigned until the checkpoint connector restores state, which is after
        every ``setup`` hook.

        Args:
            trainer: The active trainer — supplies ``ckpt_path``.
            pl_module: Unused; part of the hook signature.
        """
        super().on_train_start(trainer, pl_module)
        if self.monitor is not None or not trainer.ckpt_path or not self.dirpath:
            return
        found = sorted(
            (int(match.group(1)), str(path))
            for path in Path(self.dirpath).iterdir()
            if (match := self._rolling_pattern.fullmatch(path.name))
        )
        self._rolling = [path for _, path in found]

    def _enforce_rolling_window(self, trainer: lightning.Trainer, filepath: str) -> None:
        """Track *filepath* and drop the oldest save(s) beyond ``keep_last``.

        No-op when ``monitor`` is set — Lightning's own top-k selection owns
        retention then, unmodified.

        Args:
            trainer: The active trainer, forwarded to ``_remove_checkpoint``.
            filepath: Path just written by the caller's ``_save_checkpoint``.
        """
        if self.monitor is not None:
            return
        self._rolling.append(filepath)
        while len(self._rolling) > self.keep_last:
            self._remove_checkpoint(trainer, self._rolling.pop(0))


class _SRCheckpointBase(_RollingSaveMixin, ModelCheckpoint):
    """Everything :class:`SRCheckpoint` and :class:`SRWeightsCheckpoint` share.

    They were siblings under ``ModelCheckpoint`` rather than parent and child,
    which forced the filename construction, the rolling-mode ``save_top_k``
    rule and the monitor validation to be written twice or reached through a
    free function taking the caller's identity as an argument. They are two
    adapters over one module, so that module now exists.

    Subclasses supply defaults and, where they differ, a ``_save_checkpoint``
    payload. Nothing here decides what a checkpoint *contains*.
    """

    def _monitor_candidates(self, trainer: lightning.Trainer) -> dict[str, torch.Tensor]:
        """Substitute the batch-counted axis for ``{step}`` in every checkpoint name.

        Lightning fills ``{step}`` from ``trainer.global_step``, which counts
        *optimizer* steps. Under manual optimization with two optimizers (an
        adversarial run steps D and G per batch) that is **2x** the batch count,
        while every metric the checkpoint would be read against lands on
        ``_batches_that_stepped``. The result is an artifact whose name cannot be
        located on any curve it should be comparable with — measured before this
        override, ``experiments/SRGAN/golden`` held ``sr-400000`` against its own
        ``metrics.csv`` maxing at 199,999, while every automatic-optimization run
        was 1:1.

        Overriding here rather than at each call site is deliberate: Lightning
        routes filename formatting, top-k bookkeeping and the ``last``/``every_n``
        paths through this one method, so a single substitution keeps them
        consistent with each other. It is also invisible where the counters
        already agree, which is every SRResNet/SRCNN run.

        The name is only half of it — see :meth:`_save_checkpoint`, which records
        *both* counters in the artifact's own metadata so the axis survives the
        file being copied away from its run directory.

        Args:
            trainer: The active trainer.

        Returns:
            Lightning's candidates dict with ``step`` on the batch axis.
        """
        candidates = super()._monitor_candidates(trainer)
        candidates["step"] = torch.tensor(_logger_step(trainer))
        return candidates

    def __init__(
        self,
        monitor_metric: str | None = "psnr/val/RGB",
        save_top_k: int = 3,
        dirpath: str | None = None,
        filename_prefix: str | None = None,
        mode: str = "max",
        keep_last: int = 3,
        **kwargs: Any,
    ):
        if monitor_metric is None:
            # Lightning refuses save_top_k > 1 with nothing to rank by; the
            # window is enforced by _RollingSaveMixin instead.
            save_top_k = -1
        super().__init__(
            monitor=monitor_metric,
            mode=mode,
            save_top_k=save_top_k,
            dirpath=dirpath,
            auto_insert_metric_name=False,
            **kwargs,
        )
        self._keep_last = keep_last
        self._explicit_prefix = filename_prefix
        # The *raw* dirpath, because setup() resolves self.dirpath against the
        # trainer -- so when one callback probes another, one side may be resolved
        # and the other not, and comparing those is comparing different things.
        self._explicit_dirpath = None if dirpath is None else str(Path(dirpath).resolve())
        # A usable name before setup() can derive the real one -- direct
        # construction in a test never reaches setup.
        self._apply_naming(filename_prefix or self.DEFAULT_PREFIX)

    #: Placeholder identity used until setup() can derive the real one. Distinct
    #: per subclass so two sinks sharing a dirpath cannot collide before then.
    DEFAULT_PREFIX: str = "sr"

    def _apply_naming(self, prefix: str) -> None:
        """Set the filename template and the rolling window's matching pattern together.

        They must agree: the window finds this callback's own files by matching
        exactly what it writes, so a prefix change that touched only one of them
        would either orphan every save or adopt a sibling callback's.

        Args:
            prefix: Identity portion of the filename, without the step.
        """
        self.filename = f"{prefix}_s{{step}}"
        self._init_rolling(self._keep_last, prefix)

    def _resolved_prefix(self, pl_module: lightning.LightningModule) -> str:
        """The identity this callback will write under, whether or not setup() has run.

        Siblings are compared before Lightning has necessarily set them all up, so
        this cannot be read off ``self.filename`` -- that still holds
        ``DEFAULT_PREFIX`` on anything yet to reach setup.

        Args:
            pl_module: The model being trained; supplies the provenance a derived
                prefix is a projection of.

        Returns:
            The explicit ``filename_prefix`` if one was given, else the stem derived
            from what this callback saves.
        """
        if self._explicit_prefix is not None:
            return self._explicit_prefix
        return artifacts.stem(self._describe(pl_module))

    def _reject_filename_collision(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Raise if a sibling callback would write this callback's exact filename.

        Distinct default prefixes keep the two *classes* apart, and their docstrings
        say so -- but that reasoning never covered two instances of one class, which
        is what both shipped templates configured: one on ``psnr/val/RGB`` and one on
        ``ssim/val/RGB``, deriving the same model-provenance stem. Lightning does not
        refuse it. It keeps one and appends ``-v1`` to the other, **assigned by save
        order**, so which file wins the bare name is stable across neither runs nor
        Lightning versions, and neither name says which metric it is best on.

        The comparison is on the whole filename rather than the prefix: ``.ckpt``
        beside ``.safetensors`` under one prefix is the intended, shipped pair.

        Refusing is deliberately preferred to disambiguating automatically. Any
        automatic rule has to decide which sibling keeps the bare name, and every
        answer to that is either order-dependent -- the defect itself -- or silently
        renames published artifacts when a monitor changes.

        Args:
            trainer: The active trainer, whose ``checkpoint_callbacks`` are the
                siblings this could collide with.
            pl_module: The model being trained.

        Raises:
            MisconfigurationException: If another :class:`_SRCheckpointBase` in the
                callback list would write the same name into the same directory.
        """
        prefix = self._resolved_prefix(pl_module)
        mine = (self._explicit_dirpath, prefix, self.FILE_EXTENSION)
        for other in trainer.checkpoint_callbacks:
            if other is self or not isinstance(other, _SRCheckpointBase):
                continue
            try:
                theirs = (
                    other._explicit_dirpath,
                    other._resolved_prefix(pl_module),
                    other.FILE_EXTENSION,
                )
            except Exception:
                # A sibling whose own configuration is broken cannot be probed. Its
                # setup() reports that precisely; masking it with a collision error
                # raised from over here would be strictly worse than skipping it.
                continue
            if theirs != mine:
                continue
            raise MisconfigurationException(
                f"`{type(self).__name__}(monitor_metric={self.monitor!r})` and "
                f"`{type(other).__name__}(monitor_metric={other.monitor!r})` would both "
                f"write `{prefix}_s{{step}}{self.FILE_EXTENSION}` into the same directory. "
                "Lightning keeps one and suffixes the other `-v1`, assigned by save "
                "order, so which artifact is which is neither stable across runs nor "
                "readable from its name. HINT: give one of them a distinct "
                "`filename_prefix`."
            )

    def _describe(self, pl_module: lightning.LightningModule) -> dict[str, Any]:
        """Provenance for whatever this callback saves. Overridden per component."""
        return build_metadata(_sr(pl_module))

    def _validate_monitor(self, pl_module: lightning.LightningModule) -> None:
        """Raise if this callback's ``monitor``/``mode`` pair is unloggable or inverted.

        The direction half exists because both subclasses default to ``mode='max'``,
        which is right for PSNR/SSIM and exactly inverted for LPIPS/DISTS — the
        failure is silent, keeping the worst checkpoint of the run with nothing in
        the logs or filenames to indicate it. Which perceptual metrics are
        lower-is-better comes from ``PERCEPTUAL_METRICS`` rather than being
        restated here, so a future higher-is-better perceptual metric would still
        be checked in the right direction.

        Args:
            pl_module: The model being trained; must expose ``eval_config``.

        Raises:
            MisconfigurationException: If ``monitor`` does not name a metric
                ``SRLightning`` will log during ``fit``, or if ``mode`` disagrees
                with that metric's direction. ``monitor=None`` is rolling mode,
                which monitors nothing and is always valid.
        """
        if self.monitor is None:
            return

        eval_config = _sr(pl_module).eval_config
        higher_better = {f"psnr/val/{key}" for key in eval_config.psnr_keys}
        higher_better |= {f"ssim/val/{key}" for key in eval_config.ssim_keys}
        higher_better |= {
            f"{name}/val" for name in eval_config.perceptual_keys if not PERCEPTUAL_METRICS[name]
        }
        lower_better = {
            f"{name}/val" for name in eval_config.perceptual_keys if PERCEPTUAL_METRICS[name]
        }

        label = type(self).__name__
        valid_metrics = higher_better | lower_better
        if self.monitor not in valid_metrics:
            raise MisconfigurationException(
                f"`{label}(monitor_metric={self.monitor!r})` does not match any "
                f"metric `SRLightning` will log: {sorted(valid_metrics)}. HINT: check "
                f"`eval_config.psnr_channels` / `eval_config.separate_psnr`, "
                f"`eval_config.ssim_channels`, or `eval_config.perceptual_metrics`."
            )

        wanted = "min" if self.monitor in lower_better else "max"
        if self.mode != wanted:
            direction = "lower-is-better" if wanted == "min" else "higher-is-better"
            raise MisconfigurationException(
                f"`{label}(monitor_metric={self.monitor!r}, mode={self.mode!r})` monitors a "
                f"{direction} metric in the wrong direction — it would keep the worst "
                f"checkpoint of the run, silently. Set mode={wanted!r}."
            )

    def setup(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage: str,
    ) -> None:
        """Validate ``monitor``, derive the filename, and refuse a colliding sibling.

        Lightning itself only raises ``MisconfigurationException`` for a
        missing monitor once the val loop has already run at least once
        (``ModelCheckpoint._save_topk_checkpoint``, gated on
        ``val_loop._has_run``) — with a long ``val_check_interval`` that is a
        late, expensive-to-reach crash. This moves the same failure to startup.

        Args:
            trainer: The active trainer.
            pl_module: The model being trained; must expose ``eval_config``.
            stage: Lightning trainer stage. Only ``"fit"`` is checked — the
                monitored tags are val-loop metrics, so they are never logged
                under ``validate``/``test``/``predict`` and demanding them
                there would reject configs valid for the stage being run.

        Raises:
            MisconfigurationException: If ``monitor`` does not name a metric
                ``SRLightning`` will log during ``fit``, if ``mode`` disagrees with
                that metric's direction, or if a sibling callback would write this
                callback's exact filename.
        """
        super().setup(trainer, pl_module, stage)
        if stage != "fit":
            return
        self._validate(pl_module)
        # Derived here rather than in __init__ because it comes from the module,
        # which does not exist when a callback is built from YAML -- and after
        # _validate, because it reads whatever that just checked.
        if self._explicit_prefix is None:
            self._apply_naming(artifacts.stem(self._describe(pl_module)))
        self._reject_filename_collision(trainer, pl_module)

    def _validate(self, pl_module: lightning.LightningModule) -> None:
        """Everything that must hold before this callback can name or write anything."""
        self._validate_monitor(pl_module)


class SRCheckpoint(_SRCheckpointBase):
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
            ``None`` selects rolling mode instead: no metric is tracked at
            all, and the last ``keep_last`` checkpoints (by step) are kept.
            Needed because under an adversarial objective (e.g. SRGAN) PSNR
            and SSIM get worse by design, so top-k selection on either would
            keep the LEAST adversarial checkpoint of the run — typically one
            from the first few thousand steps.
        save_top_k: Number of best checkpoints to keep. Ignored (forced to
            ``-1``, i.e. keep-everything) when ``monitor_metric`` is ``None``
            — Lightning refuses ``save_top_k > 1`` with no monitor outright,
            so ``keep_last`` enforces the window instead.
        dirpath: Directory to save checkpoints.
        filename_prefix: Prefix for checkpoint filenames.
        keep_last: In rolling mode, number of most recent checkpoints to
            retain; oldest is deleted first. Ignored when ``monitor_metric``
            is set. Defaults to ``3``.
        **kwargs: Extra keyword arguments forwarded to
            :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    def _save_checkpoint(self, trainer: lightning.Trainer, filepath: str) -> None:
        """Save via :class:`~lightning.pytorch.callbacks.ModelCheckpoint`, then enforce rolling.

        Args:
            trainer: The active trainer.
            filepath: Destination path, already formatted by ``ModelCheckpoint``.
        """
        super()._save_checkpoint(trainer, filepath)
        # Any edit here must preserve this call -- see _RollingSaveMixin's
        # docstring for why a rolling window can't be expressed via save_top_k alone.
        self._enforce_rolling_window(trainer, filepath)


class SRWeightsCheckpoint(_SRCheckpointBase):
    """Model checkpoint that saves bare, optimizer-free weights as ``.safetensors``.

    A distributable sibling to :class:`SRCheckpoint`: that class produces resumable
    ``.ckpt`` files (full training state — optimizer moments, LR scheduler, callback
    state); this one produces ``.safetensors`` files containing only
    ``getattr(pl_module, attribute).state_dict()`` plus a matching provenance dict.
    By default ``attribute='model'`` — the bare :class:`~sisr.models.base.SRModel` (so
    keys carry no ``model.`` wrapper prefix), described by
    :func:`~sisr.training.metadata.build_metadata`. Any other ``attribute`` (e.g.
    ``'discriminator'``) saves that component instead, described by
    :func:`~sisr.training.metadata.build_component_metadata` — ``build_metadata``'s
    generator-scoped fields (``io.scale``, ``criterion``, ``eval_config``) would
    describe something a non-generator file does not contain. Both run side by side
    off the same validation metrics — top-k selection, monitor validation (inherited
    from ``SRCheckpoint``'s ``setup``), and filename formatting are inherited from
    :class:`~lightning.pytorch.callbacks.ModelCheckpoint` unchanged; only
    :meth:`_save_checkpoint` — the single method that actually writes bytes to disk —
    is overridden to swap the payload.

    ``_save_checkpoint`` is a private Lightning method, unlike the rest of this class's
    public-API surface. ``tests/training/test_callbacks.py``'s
    ``test_sr_weights_checkpoint_writes_bare_payload_via_real_fit`` guards against a
    future Lightning release routing saves through a different method: that test fails
    loudly rather than silently letting saves fall back to full, optimizer-bearing
    checkpoints under a misleadingly bare ``.safetensors`` extension.

    ``FILE_EXTENSION`` (public) and a distinct default ``filename_prefix`` keep this
    callback's top-k deletion pass — and, in rolling mode, its own oldest-first deletion
    (see :class:`_RollingSaveMixin`) — from ever touching :class:`SRCheckpoint`'s files
    when both share one ``dirpath``: each callback only ever names/deletes files matching
    its own ``filename``/``FILE_EXTENSION`` combination.

    That keeps the two *classes* apart. It does not keep two instances of one class
    apart — a pair differing only in ``monitor_metric`` derives one stem and collides,
    which :meth:`_SRCheckpointBase._reject_filename_collision` now refuses at startup.

    Args:
        monitor_metric: The validation metric to monitor — same contract as
            :class:`SRCheckpoint`, including the ``None`` rolling-mode meaning.
        save_top_k: Number of best weight files to keep. Ignored (forced to ``-1``)
            when ``monitor_metric`` is ``None`` — same contract as :class:`SRCheckpoint`.
        dirpath: Directory to save weight files.
        filename_prefix: Prefix for weight filenames. Defaults to ``'sr-weights'``,
            distinct from ``SRCheckpoint``'s ``'sr'`` default.
        mode: ``'max'`` or ``'min'`` — same contract as :class:`SRCheckpoint`.
        keep_last: In rolling mode, number of most recent weight files to retain.
            Ignored when ``monitor_metric`` is set. Defaults to ``3``.
        attribute: Name of the ``pl_module`` attribute whose ``state_dict()`` is saved.
            Defaults to ``'model'`` (the SR model). Any other value saves that named
            component instead (e.g. ``'discriminator'`` for an adversarial run's
            discriminator), with metadata scoped to that component rather than the
            generator — see the class docstring.
        **kwargs: Extra keyword arguments forwarded to
            :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    FILE_EXTENSION = artifacts.SUFFIX
    DEFAULT_PREFIX = "sr-weights"

    def __init__(
        self,
        monitor_metric: str | None = "psnr/val/RGB",
        save_top_k: int = 3,
        dirpath: str | None = None,
        filename_prefix: str | None = None,
        mode: str = "max",
        keep_last: int = 3,
        attribute: str = "model",
        **kwargs: Any,
    ):
        super().__init__(
            monitor_metric=monitor_metric,
            save_top_k=save_top_k,
            dirpath=dirpath,
            filename_prefix=filename_prefix,
            mode=mode,
            keep_last=keep_last,
            **kwargs,
        )
        self.attribute = attribute

    def _describe(self, pl_module: lightning.LightningModule) -> dict[str, Any]:
        """Component-scoped when this callback saves anything but the generator."""
        if self.attribute == "model":
            return build_metadata(_sr(pl_module))
        return build_component_metadata(_sr(pl_module), self.attribute)

    @property
    def state_key(self) -> str:
        """Lightning's callback-state key, widened by ``attribute``.

        ``ModelCheckpoint``'s own key is built from ``monitor``/``mode`` and the
        cadence fields only — not from ``attribute``, ``dirpath`` or
        ``filename``. Lightning refuses two stateful callbacks that share a key
        (``_validate_callbacks_list``), so without this, one generator and one
        discriminator weights callback on the same monitor and cadence — the
        configuration ``attribute`` exists for, and the one the SRGAN template
        ships — cannot be constructed at all.

        Appended to ``super()``'s key rather than rebuilding it, so a Lightning
        release that adds or renames a key field is carried through here
        unchanged instead of silently dropping it.

        Resuming a ``.ckpt`` written before this key gained its ``attribute``
        suffix finds no state under the new key, so this callback's monitored
        bookkeeping (``best_model_score``/``best_k_models``, i.e. what a
        ``save_top_k`` run has already kept) restarts from empty for the rest of
        that run. Rolling mode loses nothing — its window was never persisted
        under any key, and :meth:`_RollingSaveMixin.on_train_start` rebuilds it
        from this callback's own files on disk before the first save.

        Returns:
            A key unique per ``(monitor, mode, cadence, attribute)``.
        """
        return f"{super().state_key}[attribute={self.attribute}]"

    def setup(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage: str,
    ) -> None:
        """Validate ``monitor`` against the metrics ``SRLightning`` will log, and ``attribute``.

        The ``monitor`` check is inherited unchanged from
        :meth:`_SRCheckpointBase.setup`; only ``attribute`` is added here.

        ``attribute`` is otherwise only read when a checkpoint is written, so a
        typo — or ``attribute='discriminator'`` on a plain
        :class:`~sisr.training.SRLightning` — would cost a whole checkpoint
        interval before dying with a bare ``AttributeError`` from deep inside the
        save path. Same reasoning as moving the monitor check to startup.

        Args:
            trainer: The active trainer.
            pl_module: The model being trained; must expose ``eval_config`` and
                the component named by ``attribute``.
            stage: Lightning trainer stage. Only ``"fit"`` is checked — see
                ``SRCheckpoint.setup``.

        Raises:
            MisconfigurationException: If ``monitor`` does not name a metric
                ``SRLightning`` will log during ``fit``, if ``mode``
                disagrees with that metric's direction (PSNR/SSIM are
                higher-is-better; LPIPS/DISTS are lower-is-better), or if
                ``pl_module`` has no attribute named ``attribute``.
        """
        super().setup(trainer, pl_module, stage)

    def _validate(self, pl_module: lightning.LightningModule) -> None:
        """The base's monitor check, plus that ``attribute`` names something real.

        Runs before the filename is derived, which reads the very component this
        confirms exists -- otherwise a typo surfaces as a bare ``AttributeError``
        from inside the naming code rather than as this message.
        """
        super()._validate(pl_module)
        if not hasattr(pl_module, self.attribute):
            raise MisconfigurationException(
                f"`SRWeightsCheckpoint(attribute={self.attribute!r})` has nothing to save: "
                f"{type(pl_module).__name__} has no attribute {self.attribute!r}. Name a "
                f"component the module actually defines — 'model' (the generator, the "
                f"default) on any SRLightning, 'discriminator' only on an adversarial "
                f"module. Fix the callback's `attribute` in your YAML."
            )

    def _save_checkpoint(self, trainer: lightning.Trainer, filepath: str) -> None:
        """Write one component's bare weights + matching provenance metadata.

        Overrides :meth:`ModelCheckpoint._save_checkpoint` — the private hook every
        public save path (``_save_topk_checkpoint``, ``_save_none_monitor_checkpoint``,
        ``_save_last_checkpoint``) funnels through — so ``save_top_k``/filename/removal
        bookkeeping all keep working unmodified; only the on-disk payload changes.
        Mirrors the base implementation's post-save bookkeeping (``_last_global_step_saved``,
        ``_last_checkpoint_saved``, logger notification) so this callback stays a drop-in
        peer of :class:`SRCheckpoint` from the trainer's point of view.

        ``self.attribute`` selects both the saved component and its metadata builder:
        ``'model'`` (the default) uses :func:`~sisr.training.metadata.build_metadata`,
        which describes the generator; anything else uses
        :func:`~sisr.training.metadata.build_component_metadata`, so a discriminator's
        artifact never carries metadata describing a network it does not contain.

        Rolling-mode deletion (:meth:`_RollingSaveMixin._enforce_rolling_window`) runs
        last, after this method's own writes — it does not go through ``super()``
        the way :class:`SRCheckpoint`'s does, since this override never calls
        ``ModelCheckpoint._save_checkpoint`` at all (that would write the full,
        optimizer-bearing payload this class exists to avoid).

        Args:
            trainer: The active trainer — supplies ``lightning_module`` (for the
                component's ``state_dict()`` and metadata) and ``global_step``/``current_epoch``.
            filepath: Destination path, already formatted by ``ModelCheckpoint``
                (``.safetensors`` via ``FILE_EXTENSION``).
        """
        pl_module = trainer.lightning_module
        component = getattr(pl_module, self.attribute)
        monitor_value = float(self.current_score) if self.current_score is not None else None
        if self.attribute == "model":
            meta = build_metadata(
                _sr(pl_module),
                global_step=trainer.global_step,
                batch_step=_logger_step(trainer),
                epoch=trainer.current_epoch,
                monitor=self.monitor,
                monitor_value=monitor_value,
            )
        else:
            meta = build_component_metadata(
                _sr(pl_module),
                self.attribute,
                global_step=trainer.global_step,
                batch_step=_logger_step(trainer),
                epoch=trainer.current_epoch,
                monitor=self.monitor,
                monitor_value=monitor_value,
            )
        # Rank-gated to match what ModelCheckpoint's own save path gets for free:
        # it writes through Strategy.save_checkpoint, which gates on is_global_zero,
        # and this override replaces exactly that method. Without the gate every
        # process writes the same path at once. The bookkeeping below stays
        # unconditional, as it is in the base implementation, and the rolling
        # window's deletes are gated inside Strategy.remove_checkpoint already.
        if trainer.is_global_zero:
            artifacts.save(filepath, component.state_dict(), meta)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

        # Any edit here must preserve this call -- see _RollingSaveMixin's
        # docstring for why a rolling window can't be expressed via save_top_k alone.
        self._enforce_rolling_window(trainer, filepath)


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
        # Both are set for the duration of a predict run and typed optional only
        # because Lightning declares them for every stage; a hook firing without
        # them would be a Lightning bug, and saying so beats a silent skip.
        assert loader is not None, "predict hook fired with no predict dataloader"
        assert batch_indices is not None, "predict hook fired with no batch indices"
        img_paths = loader.dataset.img_paths

        for i, sample_idx in enumerate(batch_indices):
            filename = img_paths[sample_idx].stem
            out_path = self.output_dir / f"{filename}.png"
            torchvision.utils.save_image(prediction[i].clamp(0.0, 1.0), out_path)
