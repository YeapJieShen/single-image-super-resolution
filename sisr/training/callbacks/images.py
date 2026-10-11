"""Image-producing callbacks: benchmark strips and ``cli predict`` PNG output."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import lightning
import torch
import torch.nn.functional
import torchvision
from lightning.pytorch.callbacks import (
    BasePredictionWriter,
    Callback,
)

from ...metrics.scoring import Scores
from ..sinks import MetricSink, sinks_from
from ._common import _logger_step, _sr


class BenchmarkSample(NamedTuple):
    """One buffered per-image result: PSNR/SSIM only, no image tensors.

    Fields are read by name at every site, so the mean in ``_flush_buffer`` stays legible.

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


class BenchmarkImageLogger(Callback):
    """Logs bicubic|SR|HR image composites and PSNR/SSIM to TensorBoard for held-out sets.

    Fires during *both* validation and test, and the dataloader indexing
    differs between them:

    * **`cli fit` / `cli validate`** — ``val_dataloader()`` returns
      ``[primary_val] + test_loaders`` (the primary set once per scale for a multi-scale
      run), so this ignores the leading ``n_primary`` loaders (owned by
      ``SRLightning.validation_step``) and maps the rest onto ``dataset_names``. Image
      strips are throttled by ``every_n_val_runs``.
    * **`cli test`** — ``test_dataloader()`` returns the test loaders only, so
      ``0..N-1`` map straight across. No throttle; test is one-shot.

    Each image is a horizontal **bicubic | SR | HR** strip at HR scale: LR is
    bicubic-upsampled (the standard SR baseline) and SR is center-padded when
    ``padding='valid'`` made it smaller, so all three panels align.

    Tags: per-set means under ``psnr/{name}/{key}``, ``ssim/{name}/{key}`` and
    ``{metric}/{name}`` for each of ``eval_config.perceptual_keys``, every
    cycle. Per-image scalars under ``per_image/{name}/{psnr,ssim}/{key}/
    {filename}``, gated by ``log_per_image_metrics``.

    **Border cropping comes from ``pl_module.eval_config.crop_border`` at the
    use site**, never a second knob here, so the two cannot drift.

    Metrics are computed on the on-device SR/HR slices, mirroring
    ``validation_step``, and both the scalars and the strip are emitted per
    image from :meth:`_collect_batch` — **nothing image-shaped is buffered**.
    Buffering full-resolution CPU tensors until epoch end cost ~0.5 GB per
    validation cycle purely to defer ``add_image``; only a
    :class:`BenchmarkSample` is held now, for the mean in :meth:`_flush_buffer`.
    The TensorBoard experiment is resolved once in :meth:`setup`.

    Args:
        dataset_names: Ordered test set names -- the *expanded* ones for a multi-scale run
            (``Set5_x2``, ``Set5_x4``) -- matching
            ``SRDataModule.test_dataloader()`` / the trailing entries of
            ``val_dataloader()``. ``None`` auto-discovers from
            ``trainer.datamodule.test_names``.
        every_n_val_runs: Image-strip throttle for the val stage. Default
            ``5`` — with ~100 val runs that is ~20 snapshots, enough to track
            visual progress without flooding storage.
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
        self._sinks: list[MetricSink] = []  # one per attached logger that has one

    def setup(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage: str,
    ) -> None:
        """Resolve ``dataset_names``, both dataloader_idx mappings, and the logger sinks.

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
        dm = getattr(trainer, "datamodule", None)
        # The primary val set is one loader per scale, so the test loaders start after all of
        # them (1 for a single-scale run).
        n_primary = int(getattr(dm, "n_primary", 1))
        self._val_mapping = {i + n_primary: name for i, name in enumerate(self.dataset_names)}
        self._test_mapping = {i: name for i, name in enumerate(self.dataset_names)}

        self._sinks = sinks_from(trainer)

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
            self_ensemble=False,
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
            self_ensemble=_sr(pl_module).eval_config.self_ensemble,
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
        self_ensemble: bool = False,
    ) -> None:
        """Forward one batch, compute per-image metrics on-device, and stream image strips.

        Shared by :meth:`on_validation_batch_end` and
        :meth:`on_test_batch_end`. ``crop_border`` comes from
        ``pl_module.eval_config`` at the use site; *source_dataloaders*
        recovers the dataset for filename resolution.

        Metrics are computed from the on-device ``sr``/``hr_cropped`` slices, so
        only scalar ``.item()`` results leave the GPU, and **they go through
        ``pl_module.scorer``** — the object ``validation_step`` uses — rather
        than calling ``torchmetrics``/``perceptual_score`` directly, so this
        callback cannot diverge from ``validation_step`` on the crop, the
        reductions, or which backbone an ``eval_config`` means.

        ``self_ensemble`` is forwarded to ``predict_rgb`` unchanged: :meth:`on_test_batch_end`
        passes ``pl_module.eval_config.self_ensemble``; :meth:`on_validation_batch_end` always
        passes ``False`` — self-ensembling must never apply during validation, whether that
        validation is `fit`'s periodic cycle or a bare `cli validate`.

        When images or per-image scalars are wanted, exactly one host transfer
        per image composes them and both are emitted immediately — **nothing
        image-shaped is buffered**; only a ``BenchmarkSample`` is kept, for
        :meth:`_flush_buffer`'s per-set mean.
        """
        lr_img, hr_img = batch[0], batch[1]
        is_grey = batch[2] if len(batch) > 2 else None

        with torch.no_grad():
            sr, hr_cropped = _sr(pl_module).predict_rgb(lr_img, hr_img, self_ensemble=self_ensemble)

        # Resolve filenames from the underlying dataset for use as TB tags.
        # Optional only because Lightning declares the loader lists for every
        # stage; the hooks that call this one fire only while theirs is set.
        assert source_dataloaders is not None, "no dataloaders for the running stage"
        loader = source_dataloaders[dataloader_idx]
        dataset = loader.dataset
        loader_batch_size = loader.batch_size
        current_batch_size = lr_img.size(0)
        # Resolved once per batch, and only when something will actually be
        # emitted — a trainer with no fit loop is legitimate when no TB logger
        # is attached, and the guard below already skips every emission then.
        sinks = self._sinks
        step = _logger_step(trainer) if sinks else 0

        for i in range(current_batch_size):
            global_idx = batch_idx * loader_batch_size + i
            filename = dataset.img_paths[global_idx].stem

            # One scorer object, shared with validation_step, so the two paths cannot
            # disagree on the crop, the colorspace split, the PSNR reduction (per image,
            # never whole-batch pooled) or which SSIM eval_config.ssim_impl names.
            scores = _sr(pl_module).scorer.score(
                sr[i : i + 1],
                hr_cropped[i : i + 1],
                scale=getattr(dataset, "scale", None),
                is_grey=None if is_grey is None else is_grey[i : i + 1],
            )
            # float(), not .item(): Scores carries either tensors (this path,
            # straight from the scorer) or plain floats (the means _flush_buffer
            # builds). Exact for a 0-dim tensor, and total over both.
            psnr_dict = {key: float(value) for key, value in scores.psnr.items()}
            ssim_dict = {key: float(value) for key, value in scores.ssim.items()}
            perceptual_dict = {name: float(value) for name, value in scores.perceptual.items()}
            self._buffer[dataset_name].append(
                BenchmarkSample(filename, psnr_dict, ssim_dict, perceptual_dict)
            )

            if not sinks:
                continue

            if self.log_per_image_metrics:
                self._emit_per_image_scalars(
                    sinks, dataset_name, filename, psnr_dict, ssim_dict, step
                )

            if not should_log_images:
                continue

            # One host transfer per image, only on cycles that log strips.
            self._emit_triptych(
                sinks, dataset_name, filename, lr_img[i].cpu(), sr[i].cpu(), hr_img[i].cpu(), step
            )

    def _emit_per_image_scalars(
        self,
        sinks: list[MetricSink],
        dataset_name: str,
        filename: str,
        psnr: dict[str, float],
        ssim: dict[str, float],
        step: int,
    ) -> None:
        """Write one image's PSNR/SSIM scalars under the ``per_image/`` hierarchy.

        A deeper hierarchy than the aggregate tags on purpose: these are one
        series per image per key, and mixing them into ``psnr/{set}/{key}``
        would bury the per-set means they are meant to explain.

        Args:
            sinks: Where to write.
            dataset_name: Benchmark set the image belongs to.
            filename: Source image stem, the tag's leaf.
            psnr: ``key -> value`` for this one image.
            ssim: ``key -> value`` for this one image.
            step: Logger step to stamp, from ``_logger_step``.
        """
        for sink in sinks:
            for key, value in psnr.items():
                sink.scalar(f"per_image/{dataset_name}/psnr/{key}/{filename}", value, step)
            for key, value in ssim.items():
                sink.scalar(f"per_image/{dataset_name}/ssim/{key}/{filename}", value, step)

    def _emit_triptych(
        self,
        sinks: list[MetricSink],
        dataset_name: str,
        filename: str,
        lr_cpu: torch.Tensor,
        sr_cpu: torch.Tensor,
        hr_cpu: torch.Tensor,
        step: int,
    ) -> None:
        """Compose and emit one ``bicubic | SR | HR`` strip, all panels at HR size.

        The bicubic panel is the LR upsampled to HR by bicubic interpolation --
        the standard SR baseline. For SRCNN the LR is already at HR size
        (pre-upsampled in the dataset), so this resamples at 1:1 and is
        near-identity; for SRResNet (LR < HR) it upscales for direct visual
        comparison against SR. SR is centre-padded **only when smaller than
        HR** (SRCNN with ``padding='valid'``), which preserves full HR context
        on the right rather than cropping the reference to match.

        Args:
            sinks: Where to write.
            dataset_name: Benchmark set, the tag's first segment.
            filename: Source image stem, the tag's leaf.
            lr_cpu: This image's LR, already on the host.
            sr_cpu: This image's reconstruction, already on the host.
            hr_cpu: This image's reference, already on the host.
            step: Logger step to stamp, from ``_logger_step``.
        """
        target_hw = hr_cpu.shape[-2:]
        bicubic = self._bicubic_to(lr_cpu, target_hw)
        sr_padded = self._pad_to_match(sr_cpu, target_hw)
        strip = torchvision.utils.make_grid(
            [bicubic, sr_padded, hr_cpu], nrow=3, padding=2, pad_value=0.5
        )
        for sink in sinks:
            sink.image(f"{dataset_name}/{filename}", strip, step)

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

            # Mean each family, then let Scores.tagged apply the grammar --
            # the same call validation_step's own emission goes through. The
            # perceptual family's different tag shape comes along for free,
            # which is the part a hand-rolled f-string kept getting to own.
            n = len(samples)
            means = Scores(
                psnr={k: sum(s.psnr[k] for s in samples) / n for k in samples[0].psnr},
                ssim={k: sum(s.ssim[k] for s in samples) / n for k in samples[0].ssim},
                perceptual={
                    k: sum(s.perceptual[k] for s in samples) / n for k in samples[0].perceptual
                },
            )
            for tag, mean in means.tagged(dataset_name).items():
                pl_module.log(tag, mean, add_dataloader_idx=False, sync_dist=True)

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
