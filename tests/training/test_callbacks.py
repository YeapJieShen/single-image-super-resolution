import functools
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import lightning
import numpy as np
import pytest
import torch
import torchmetrics.functional
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from PIL import Image

from sisr.models.srcnn import SRCNN
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import (
    BenchmarkImageLogger,
    GradNormLogger,
    SRCheckpoint,
    SREvalConfig,
    SRLightning,
    SRPredictionWriter,
    SRTrainingConfig,
    SRWeightsCheckpoint,
    WeightHistogramLogger,
)
from sisr.training.callbacks import BenchmarkSample

# ---------------------------------------------------------------------------
# BenchmarkImageLogger.setup auto-discovery
# ---------------------------------------------------------------------------


def test_benchmark_setup_auto_discovers_dataset_names():
    cb = BenchmarkImageLogger()
    trainer = SimpleNamespace(datamodule=SimpleNamespace(test_names=["Set5", "Set14", "Urban100"]))
    cb.setup(trainer, pl_module=None, stage="fit")
    assert cb.dataset_names == ["Set5", "Set14", "Urban100"]
    assert cb._val_mapping == {1: "Set5", 2: "Set14", 3: "Urban100"}
    assert cb._test_mapping == {0: "Set5", 1: "Set14", 2: "Urban100"}


def test_benchmark_setup_explicit_names_skip_auto_discovery():
    cb = BenchmarkImageLogger(dataset_names=["A", "B"])
    trainer = SimpleNamespace(datamodule=SimpleNamespace(test_names=["X", "Y", "Z"]))
    cb.setup(trainer, pl_module=None, stage="fit")
    assert cb.dataset_names == ["A", "B"]
    assert cb._val_mapping == {1: "A", 2: "B"}


def test_benchmark_setup_no_datamodule_results_in_empty_mapping():
    cb = BenchmarkImageLogger()
    trainer = SimpleNamespace(datamodule=None)
    cb.setup(trainer, pl_module=None, stage="fit")
    assert cb.dataset_names == []
    assert cb._val_mapping == {}
    assert cb._test_mapping == {}


def test_benchmark_setup_datamodule_without_test_names_is_safe():
    cb = BenchmarkImageLogger()
    trainer = SimpleNamespace(datamodule=SimpleNamespace())  # no test_names attr
    cb.setup(trainer, pl_module=None, stage="fit")
    assert cb.dataset_names == []


def _make_step_axis_trainer(
    global_step: int, batches_that_stepped: int, loggers: list | None = None
) -> SimpleNamespace:
    """Trainer stub whose two step axes disagree, as they do under manual optimization.

    SRGAN steps D and G per batch, so ``global_step`` is 2x the batch count while
    Lightning keeps ``_batches_that_stepped`` batch-counted "disregarding multiple
    optimizers on purpose for loggers" (``loops/training_epoch_loop.py``). A stub
    where the two agree passes whichever axis the callback reads, so it observes
    nothing — the disagreement is the entire point of this fixture.
    """
    return SimpleNamespace(
        global_step=global_step,
        fit_loop=SimpleNamespace(
            epoch_loop=SimpleNamespace(_batches_that_stepped=batches_that_stepped)
        ),
        loggers=[] if loggers is None else loggers,
    )


def _make_benchmark_pl_module() -> MagicMock:
    """Stub module exercising only ``_collect_batch``'s PSNR path."""
    pl_module = MagicMock()
    pl_module.eval_config = SimpleNamespace(
        crop_border=0, psnr_keys=["RGB"], ssim_keys=[], perceptual_keys=[]
    )
    pl_module.predict_rgb = MagicMock(return_value=(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)))
    pl_module._build_metric_tensors = MagicMock(side_effect=lambda s, h: {"RGB": (s, h)})
    return pl_module


def test_benchmark_logs_on_the_batch_counted_axis_not_global_step():
    """Image strips and per-image scalars must share the axis ``self.log`` uses.

    Logging at ``trainer.global_step`` puts every benchmark image at twice the
    step of the loss curves it is read against, for any module training under
    manual optimization with two optimizers.
    """
    cb = BenchmarkImageLogger(log_per_image_metrics=True)
    cb._buffer["Set5"] = []
    cb._tb_experiment = MagicMock()
    trainer = _make_step_axis_trainer(global_step=400, batches_that_stepped=200)
    dataloaders = [SimpleNamespace(dataset=SimpleNamespace(img_paths=[Path("baby.png")]))]

    cb._collect_batch(
        trainer,
        _make_benchmark_pl_module(),
        batch=(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)),
        batch_idx=0,
        dataset_name="Set5",
        source_dataloaders=dataloaders,
        dataloader_idx=0,
        should_log_images=True,
    )

    assert cb._tb_experiment.add_image.call_args.kwargs["global_step"] == 200
    assert cb._tb_experiment.add_scalar.call_args.kwargs["global_step"] == 200


def test_benchmark_psnr_routes_through_the_module_not_torchmetrics_directly():
    """The benchmark path must score PSNR with the same reduction as validation.

    ``SRLightning._mean_psnr`` uses ``dim=(1,2,3)`` — per-image PSNR then batch
    mean — and documents that reduction as deliberate. ``_collect_batch`` used
    to call ``torchmetrics`` directly with the default ``dim=None``, which pools
    the whole batch into one PSNR. The two agree today only because this path
    slices ``sr[i:i+1]``, so every call happens to be a batch of one; nothing
    stated or tested that precondition, and batching this loop would have
    silently changed every ``psnr/{set}/{key}`` number.

    Asserted through the emitted value rather than by spying on the call: a
    stub ``_mean_psnr`` returns a sentinel, and the buffered sample must carry
    it. A callback computing its own PSNR cannot produce the sentinel, so this
    goes red against the direct-torchmetrics version without caring *how* the
    module is reached.
    """
    cb = BenchmarkImageLogger()
    cb._buffer["Set5"] = []
    pl_module = _make_benchmark_pl_module()
    pl_module._mean_psnr = MagicMock(return_value=torch.tensor(42.0))
    dataloaders = [SimpleNamespace(dataset=SimpleNamespace(img_paths=[Path("baby.png")]))]

    cb._collect_batch(
        trainer=_make_step_axis_trainer(global_step=2, batches_that_stepped=1),
        pl_module=pl_module,
        batch=(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)),
        batch_idx=0,
        dataset_name="Set5",
        source_dataloaders=dataloaders,
        dataloader_idx=0,
        should_log_images=False,
    )

    assert cb._buffer["Set5"][0].psnr["RGB"] == pytest.approx(42.0)


def test_the_two_psnr_reductions_genuinely_differ_above_batch_size_one():
    """Characterises the dependency the test above exists to guard.

    Not a test of our code: it pins torchmetrics' behaviour that makes the
    routing matter. If these two ever coincide at batch > 1, the invariant is
    free and the guard above is redundant — this test is what would tell us.
    """
    # The per-image errors must DIFFER: 0.05 and 0.40. Two images with the same
    # error give equal per-image PSNRs, and pooling then agrees with the mean --
    # which is how the first draft of this test passed against both reductions.
    sr = torch.stack([torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)])
    hr = torch.stack([torch.full((3, 8, 8), 0.05), torch.full((3, 8, 8), 0.40)])

    pooled = torchmetrics.functional.image.peak_signal_noise_ratio(sr, hr, data_range=1.0)
    per_image = torchmetrics.functional.image.peak_signal_noise_ratio(
        sr, hr, data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean"
    )
    assert pooled.item() != pytest.approx(per_image.item())


# ---------------------------------------------------------------------------
# BenchmarkImageLogger._bicubic_to
# ---------------------------------------------------------------------------


def test_bicubic_to_returns_target_shape():
    """LR (3, 4, 4) -> (3, 16, 16) for a 4x upscaling model."""
    lr = torch.rand(3, 4, 4)
    out = BenchmarkImageLogger._bicubic_to(lr, (16, 16))
    assert out.shape == (3, 16, 16)


def test_bicubic_to_clamps_to_unit_interval():
    """Bicubic overshoots on high-contrast inputs; the helper clamps so the
    panel renders cleanly as a normalized image."""
    lr = torch.zeros(3, 4, 4)
    lr[:, 0, 0] = 1.0
    out = BenchmarkImageLogger._bicubic_to(lr, (16, 16))
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_bicubic_to_matches_torch_interpolate_bicubic():
    """Lock the helper to bicubic mode (not bilinear/nearest)."""
    torch.manual_seed(0)
    lr = torch.rand(3, 4, 4)
    expected = (
        torch.nn.functional.interpolate(
            lr.unsqueeze(0), size=(16, 16), mode="bicubic", align_corners=False
        )
        .squeeze(0)
        .clamp(0.0, 1.0)
    )
    out = BenchmarkImageLogger._bicubic_to(lr, (16, 16))
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# BenchmarkImageLogger._pad_to_match
# ---------------------------------------------------------------------------


def test_pad_to_match_no_op_when_shapes_match():
    img = torch.zeros(3, 8, 8)
    out = BenchmarkImageLogger._pad_to_match(img, (8, 8))
    assert out.shape == (3, 8, 8)
    assert torch.equal(out, img)


def test_pad_to_match_symmetric_pad():
    img = torch.ones(3, 6, 6)
    out = BenchmarkImageLogger._pad_to_match(img, (8, 8))
    assert out.shape == (3, 8, 8)
    # Border zero, inner 6x6 still ones.
    assert torch.equal(out[:, 1:7, 1:7], img)
    assert (out[:, 0, :] == 0).all()
    assert (out[:, :, 0] == 0).all()


def test_pad_to_match_off_by_one_height_only():
    """target=(9, 8), input=(8, 8): only height is off by 1.  After symmetric
    pad the shape stays (8, 8); the corrective branch must fire to add the
    final row, giving (9, 8)."""
    img = torch.zeros(3, 8, 8)
    out = BenchmarkImageLogger._pad_to_match(img, (9, 8))
    assert out.shape == (3, 9, 8)


def test_pad_to_match_off_by_one_width_only():
    img = torch.zeros(3, 8, 8)
    out = BenchmarkImageLogger._pad_to_match(img, (8, 9))
    assert out.shape == (3, 8, 9)


def test_pad_to_match_off_by_one_both_dims():
    img = torch.zeros(3, 8, 8)
    out = BenchmarkImageLogger._pad_to_match(img, (9, 9))
    assert out.shape == (3, 9, 9)


# ---------------------------------------------------------------------------
# GradNormLogger
# ---------------------------------------------------------------------------


def _make_gradnorm_pl_module():
    """Stub LightningModule with parameters that have non-zero grads."""
    pl_module = MagicMock()
    p1 = torch.zeros(4, requires_grad=True)
    p1.grad = torch.ones(4) * 3.0  # norm = sqrt(4 * 9) = 6
    p2 = torch.zeros(4, requires_grad=True)
    p2.grad = torch.ones(4) * 4.0  # norm = sqrt(4 * 16) = 8
    # Total: sqrt(36 + 64) = sqrt(100) = 10
    pl_module.parameters = MagicMock(return_value=[p1, p2])
    return pl_module


def test_grad_norm_logger_logs_on_cadence():
    cb = GradNormLogger(log_every_n_steps=10)
    # batch axis ON cadence, optimizer axis OFF -- the gate must read the former.
    trainer = _make_step_axis_trainer(global_step=17, batches_that_stepped=10)
    pl_module = _make_gradnorm_pl_module()
    cb.on_after_backward(trainer, pl_module)
    pl_module.log.assert_called_once()
    args, _ = pl_module.log.call_args
    assert args[0] == "diag/grad_norm"
    assert args[1] == pytest.approx(10.0)


def test_grad_norm_logger_skips_off_cadence():
    cb = GradNormLogger(log_every_n_steps=10)
    # batch axis OFF cadence, optimizer axis ON.
    trainer = _make_step_axis_trainer(global_step=20, batches_that_stepped=7)
    pl_module = _make_gradnorm_pl_module()
    cb.on_after_backward(trainer, pl_module)
    pl_module.log.assert_not_called()


def test_grad_norm_logger_handles_none_grads():
    cb = GradNormLogger(log_every_n_steps=1)
    pl_module = MagicMock()
    p = torch.zeros(4, requires_grad=True)
    p.grad = None
    pl_module.parameters = MagicMock(return_value=[p])
    trainer = _make_step_axis_trainer(global_step=2, batches_that_stepped=1)
    cb.on_after_backward(trainer, pl_module)
    args, _ = pl_module.log.call_args
    assert args[1] == 0.0


def test_grad_norm_logger_uses_grad_detach_not_grad_data():
    """Gradient-norm accumulation must read gradients via
    ``p.grad.detach()``, not the legacy ``p.grad.data`` attribute — and must
    still compute the correct L2 norm. The source check gives the red->green
    signal (``.grad.data`` does not raise a Python warning on the pinned torch,
    so a warnings-based assert cannot go red); the numeric check confirms
    behaviour is unchanged."""
    import inspect

    src = inspect.getsource(GradNormLogger.on_after_backward)
    assert ".grad.data" not in src
    assert "p.grad.detach()" in src

    cb = GradNormLogger(log_every_n_steps=1)
    trainer = _make_step_axis_trainer(global_step=2, batches_that_stepped=1)
    pl_module = _make_gradnorm_pl_module()
    cb.on_after_backward(trainer, pl_module)
    args, _ = pl_module.log.call_args
    assert args[0] == "diag/grad_norm"
    assert args[1] == pytest.approx(10.0)


def test_grad_norm_logger_cadence_counts_batches_not_optimizer_steps():
    """``log_every_n_steps`` must gate on the axis ``self.log`` writes to.

    ``trainer.global_step`` counts *optimizer* steps, so under manual
    optimization gating on it makes the configured cadence mean a different
    number of batches per training paradigm, and leaves ``diag/grad_norm``
    sampled against one counter while plotted against another.

    Both directions are asserted deliberately: a stub whose two axes agree, or a
    single direction, passes under a gate reading either axis.
    """
    # global_step ON cadence, batch count OFF -> must not fire.
    cb = GradNormLogger(log_every_n_steps=10)
    pl_module = _make_gradnorm_pl_module()
    cb.on_after_backward(_make_step_axis_trainer(global_step=10, batches_that_stepped=7), pl_module)
    pl_module.log.assert_not_called()

    # global_step OFF cadence, batch count ON -> must fire.
    cb = GradNormLogger(log_every_n_steps=10)
    pl_module = _make_gradnorm_pl_module()
    cb.on_after_backward(
        _make_step_axis_trainer(global_step=14, batches_that_stepped=10), pl_module
    )
    pl_module.log.assert_called_once()


# ---------------------------------------------------------------------------
# WeightHistogramLogger
# ---------------------------------------------------------------------------


def _make_histogram_probe(tmp_path: Path, monkeypatch):
    """Real ``TensorBoardLogger`` with ``add_histogram`` spied, plus a stub module.

    A real logger, not a mock: the callback reaches ``tb_logger.experiment``
    through an ``isinstance`` check on ``trainer.loggers``, so a bare mock would
    be filtered out and every assertion below would pass vacuously.
    """
    import lightning.pytorch.loggers as pl_loggers

    pl_module = MagicMock()
    pl_module.named_parameters = MagicMock(
        return_value=[("model.feat.0.weight", torch.nn.Parameter(torch.rand(4, 3, 3, 3)))]
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    add_histogram = MagicMock(wraps=tb_logger.experiment.add_histogram)
    monkeypatch.setattr(tb_logger.experiment, "add_histogram", add_histogram)
    return pl_module, tb_logger, add_histogram


def test_weight_histogram_logger_cadence_counts_batches_not_optimizer_steps(
    tmp_path: Path, monkeypatch
):
    """Same gate, same reason as the ``GradNormLogger`` case above."""
    pl_module, tb_logger, add_histogram = _make_histogram_probe(tmp_path, monkeypatch)
    cb = WeightHistogramLogger(log_every_n_steps=10)

    # global_step ON cadence, batch count OFF -> must not fire.
    trainer = _make_step_axis_trainer(global_step=10, batches_that_stepped=7)
    trainer.loggers = [tb_logger]
    cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
    add_histogram.assert_not_called()

    # global_step OFF cadence, batch count ON -> must fire.
    trainer = _make_step_axis_trainer(global_step=14, batches_that_stepped=10)
    trainer.loggers = [tb_logger]
    cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
    add_histogram.assert_called_once()
    tb_logger.finalize("success")


def test_weight_histogram_logger_writes_on_the_batch_counted_axis_not_global_step(
    tmp_path: Path, monkeypatch
):
    """Histograms go to the raw TB experiment, so they must carry the batch axis.

    This is the same defect fixed for benchmark images: writing at
    ``trainer.global_step`` puts every histogram at twice the step of the loss
    curves it is read against, for any module under manual optimization with two
    optimizers.
    """
    pl_module, tb_logger, add_histogram = _make_histogram_probe(tmp_path, monkeypatch)
    cb = WeightHistogramLogger(log_every_n_steps=1)

    trainer = _make_step_axis_trainer(global_step=400, batches_that_stepped=200)
    trainer.loggers = [tb_logger]
    cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
    tb_logger.finalize("success")

    add_histogram.assert_called_once()
    assert add_histogram.call_args.kwargs["global_step"] == 200


def test_weight_histogram_logger_skips_off_cadence():
    cb = WeightHistogramLogger(log_every_n_steps=10)
    trainer = _make_step_axis_trainer(global_step=20, batches_that_stepped=7)
    pl_module = MagicMock()
    cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
    pl_module.named_parameters.assert_not_called()


def test_weight_histogram_logger_skips_when_no_tb_logger():
    cb = WeightHistogramLogger(log_every_n_steps=1)
    trainer = _make_step_axis_trainer(global_step=2, batches_that_stepped=1)
    pl_module = MagicMock()
    cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
    pl_module.named_parameters.assert_not_called()


# ---------------------------------------------------------------------------
# SRCheckpoint
# ---------------------------------------------------------------------------


def test_srcheckpoint_filename_pattern():
    """monitor_metric='val_psnr(RGB)' -> filename pattern uses that metric."""
    ckpt = SRCheckpoint(monitor_metric="val_psnr(RGB)", save_top_k=3, dirpath="/tmp/x")
    # ModelCheckpoint stores `filename` — verify the metric appears in the format.
    assert "val_psnr(RGB)" in ckpt.filename
    assert "step" in ckpt.filename


def test_srcheckpoint_default_max_mode():
    ckpt = SRCheckpoint(monitor_metric="val_psnr(RGB)", save_top_k=3, dirpath="/tmp/x")
    assert ckpt.mode == "max"


def test_srcheckpoint_custom_filename_prefix():
    ckpt = SRCheckpoint(
        monitor_metric="val_ssim",
        save_top_k=3,
        dirpath="/tmp/x",
        filename_prefix="myrun",
    )
    assert ckpt.filename.startswith("myrun-")


def test_srcheckpoint_default_filename_prefix_is_neutral_sr():
    """The generic checkpoint's default prefix must be arch-neutral 'sr',
    not the SRCNN-specific 'srcnn'."""
    ckpt = SRCheckpoint(monitor_metric="val_psnr(RGB)", dirpath="/tmp/x")
    assert ckpt.filename.startswith("sr-")


def test_srcheckpoint_disables_auto_insert_metric_name():
    """auto_insert_metric_name must be off, since it is the
    mechanism that would otherwise splice the raw (possibly `/`-bearing) metric
    name into the filename as literal text."""
    ckpt = SRCheckpoint(monitor_metric="val_psnr(RGB)", dirpath="/tmp/x")
    assert ckpt.auto_insert_metric_name is False


def test_srcheckpoint_slash_monitor_renders_flat_filename(tmp_path: Path):
    """A `/`-bearing monitor_metric (e.g. a future
    `psnr/val/RGB` TensorBoard-hierarchy tag) must render to a flat basename
    with no path separator. Lightning's `_format_checkpoint_name` does no
    sanitisation of its own — with the default `auto_insert_metric_name=True`
    it splices the raw metric name in as literal text before the value
    placeholder, which would make `TorchCheckpointIO.save_checkpoint`
    silently create a nested directory tree (one per save) instead of a flat
    checkpoint file. The `metrics` dict lookup that supplies the *value* is
    unaffected by the `/` — it's just a dict key."""
    ckpt = SRCheckpoint(monitor_metric="psnr/val/RGB", dirpath=str(tmp_path))
    rendered = ckpt.format_checkpoint_name(
        {"step": torch.tensor(1000), "psnr/val/RGB": torch.tensor(30.1234)}
    )
    basename = Path(rendered).name
    assert "/" not in basename
    assert basename == "sr-1000-psnr_val_RGB=30.1234.ckpt"


# ---------------------------------------------------------------------------
# SRCheckpoint.setup monitor validation
# ---------------------------------------------------------------------------


def _make_bare_trainer() -> lightning.Trainer:
    """Minimal, unfitted Trainer — enough for ModelCheckpoint.setup's
    `__resolve_ckpt_dir` / `trainer.strategy.broadcast` / `is_global_zero`
    calls without running an actual loop."""
    return lightning.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


# _make_bare_trainer's accelerator="cpu" (matching CI, which has no GPU) logs a
# "GPU available but not used" warning on a CUDA dev box; harmless, but the
# strict global filterwarnings=error would fail on it otherwise.
_ignore_gpu_warning = pytest.mark.filterwarnings(
    "ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning"
)


@_ignore_gpu_warning
def test_srcheckpoint_setup_accepts_monitor_matching_psnr_keys(tmp_path: Path):
    """monitor_metric derived from eval_config.psnr_keys must pass setup()
    without raising."""
    pl_module = _make_real_pl_module()  # SREvalConfig(crop_border=0) -> psnr_keys=['RGB']
    ckpt = SRCheckpoint(monitor_metric="psnr/val/RGB", dirpath=str(tmp_path))
    ckpt.setup(_make_bare_trainer(), pl_module, stage="fit")  # must not raise


@_ignore_gpu_warning
def test_srcheckpoint_setup_accepts_val_ssim_monitor(tmp_path: Path):
    """ssim/val/{key} for any key in eval_config.ssim_keys must be accepted."""
    pl_module = _make_real_pl_module()  # ssim_channels defaults to ['RGB', 'Y']
    ckpt = SRCheckpoint(monitor_metric="ssim/val/RGB", dirpath=str(tmp_path))
    ckpt.setup(_make_bare_trainer(), pl_module, stage="fit")  # must not raise


@_ignore_gpu_warning
def test_srcheckpoint_setup_rejects_monitor_not_in_psnr_keys(tmp_path: Path):
    """Monitoring a PSNR key eval_config never requests (here 'Y',
    since psnr_channels=['RGB'] and separate_psnr=False) must raise
    MisconfigurationException at setup() time — startup, not 20k steps into
    training once Lightning's own val_loop._has_run-gated check would fire."""
    pl_module = _make_real_pl_module()
    ckpt = SRCheckpoint(monitor_metric="psnr/val/Y", dirpath=str(tmp_path))
    with pytest.raises(MisconfigurationException, match=r"psnr/val/Y"):
        ckpt.setup(_make_bare_trainer(), pl_module, stage="fit")


@_ignore_gpu_warning
def test_srcheckpoint_setup_error_lists_valid_metrics(tmp_path: Path):
    """The error message must enumerate the actually-loggable metric set, so a
    misconfigured monitor is actionable without reading source."""
    pl_module = _make_real_pl_module()
    ckpt = SRCheckpoint(monitor_metric="bogus_metric", dirpath=str(tmp_path))
    with pytest.raises(MisconfigurationException) as exc_info:
        ckpt.setup(_make_bare_trainer(), pl_module, stage="fit")
    assert "psnr/val/RGB" in str(exc_info.value)
    assert "ssim/val/RGB" in str(exc_info.value)
    assert "ssim/val/Y" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _SRCheckpointBase._validate_monitor direction check (LPIPS/DISTS are lower-is-better)
# ---------------------------------------------------------------------------


def build_module(eval_config: SREvalConfig | None = None) -> SRLightning:
    """Real SRLightning exposing only the `eval_config` `_validate_monitor` reads."""
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=eval_config or SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def test_perceptual_monitor_is_accepted():
    module = build_module(eval_config=SREvalConfig(perceptual_metrics=["lpips"]))
    SRCheckpoint(monitor_metric="lpips/val", mode="min")._validate_monitor(module)  # no raise


def test_lower_is_better_metric_rejects_mode_max():
    """SRCheckpoint defaults mode='max'; LPIPS and DISTS are lower-better.

    Monitoring lpips/val at the default keeps the WORST model for the whole
    run, and nothing in the logs, tags or filenames says so.
    """
    module = build_module(eval_config=SREvalConfig(perceptual_metrics=["lpips"]))
    with pytest.raises(MisconfigurationException, match="lower-is-better"):
        SRCheckpoint(monitor_metric="lpips/val", mode="max")._validate_monitor(module)


def test_psnr_monitor_still_requires_mode_max():
    module = build_module(eval_config=SREvalConfig())
    with pytest.raises(MisconfigurationException, match="higher-is-better"):
        SRCheckpoint(monitor_metric="psnr/val/RGB", mode="min")._validate_monitor(module)


@pytest.mark.parametrize("cls", [SRCheckpoint, SRWeightsCheckpoint])
def test_monitor_error_names_the_actual_callback_class(cls):
    """The misconfiguration message must name the class that was misconfigured.

    The label used to be a hard-coded literal at each call site; it is now
    derived from ``type(self).__name__`` so the two cannot drift. Nothing
    pinned it, so deriving it wrongly — or reverting to a literal on the base,
    which would name every subclass identically — was invisible to the suite.
    Both classes are asserted because a single-class check passes under a
    hard-coded literal that happens to match that one class.
    """
    module = build_module()
    cb = cls(monitor_metric="not/a/metric", mode="max")
    with pytest.raises(MisconfigurationException, match=cls.__name__):
        cb._validate_monitor(module)


# ---------------------------------------------------------------------------
# SRWeightsCheckpoint
# ---------------------------------------------------------------------------


def test_sr_weights_checkpoint_file_extension_is_pt():
    ckpt = SRWeightsCheckpoint(monitor_metric="psnr/val/RGB", dirpath="/tmp/x")
    assert ckpt.FILE_EXTENSION == ".pt"


def test_sr_weights_checkpoint_default_filename_prefix_is_sr_weights():
    """Distinct from SRCheckpoint's 'sr' default so a shared dirpath's top-k
    deletion passes can never mistake one callback's files for the other's."""
    ckpt = SRWeightsCheckpoint(monitor_metric="psnr/val/RGB", dirpath="/tmp/x")
    assert ckpt.filename.startswith("sr-weights-")


def test_sr_weights_checkpoint_custom_filename_prefix():
    ckpt = SRWeightsCheckpoint(
        monitor_metric="psnr/val/RGB", dirpath="/tmp/x", filename_prefix="myrun"
    )
    assert ckpt.filename.startswith("myrun-")


def test_sr_weights_checkpoint_default_max_mode():
    ckpt = SRWeightsCheckpoint(monitor_metric="psnr/val/RGB", dirpath="/tmp/x")
    assert ckpt.mode == "max"


@_ignore_gpu_warning
def test_sr_weights_checkpoint_setup_rejects_unknown_monitor(tmp_path: Path):
    """setup() validation is inherited (delegated to SRCheckpoint.setup), so a
    monitor eval_config never logs must raise at startup here too."""
    pl_module = _make_real_pl_module()
    ckpt = SRWeightsCheckpoint(monitor_metric="psnr/val/Y", dirpath=str(tmp_path))
    with pytest.raises(MisconfigurationException, match=r"psnr/val/Y"):
        ckpt.setup(_make_bare_trainer(), pl_module, stage="fit")


@_ignore_gpu_warning
def test_sr_weights_checkpoint_setup_rejects_unknown_attribute(tmp_path: Path):
    """A component this module does not have must fail at startup, not at the
    first save.

    ``attribute`` is read nowhere else, so a typo — or ``discriminator`` on a
    plain ``SRLightning``, which is a one-line YAML mistake — otherwise survives
    a whole ``every_n_train_steps`` interval and then dies with a bare
    ``AttributeError`` from inside the save path.
    """
    ckpt = SRWeightsCheckpoint(
        monitor_metric=None, attribute="discriminator", dirpath=str(tmp_path)
    )
    with pytest.raises(MisconfigurationException, match=r"discriminator"):
        ckpt.setup(_make_bare_trainer(), build_module(), stage="fit")


@_ignore_gpu_warning
def test_sr_weights_checkpoint_setup_accepts_a_present_component(tmp_path: Path):
    """The refusal above must not reject the configuration ``attribute`` exists for."""
    ckpt = SRWeightsCheckpoint(
        monitor_metric=None, attribute="discriminator", dirpath=str(tmp_path)
    )
    ckpt.setup(_make_bare_trainer(), build_module_with_component(), stage="fit")  # must not raise


def test_sr_weights_checkpoint_save_checkpoint_is_actually_overridden():
    """Cheap static guard: the override must exist as a distinct method, not
    silently inherit ModelCheckpoint's (which would write full, optimizer-bearing
    checkpoints under the .pt extension). The real, functional guard against a
    Lightning release routing saves through a different private method entirely
    is test_sr_weights_checkpoint_writes_bare_payload_via_real_fit below, which
    exercises the actual save path end-to-end."""
    assert SRWeightsCheckpoint._save_checkpoint is not ModelCheckpoint._save_checkpoint


class _StubComponent(torch.nn.Module):
    """Stand-in for a not-yet-existing named component (e.g. a discriminator).

    Deliberately not named after any real architecture -- SRDiscriminator
    doesn't exist until a later task, and asserting metadata against a
    hardcoded "SRDiscriminator" string here would test a coincidence of
    naming, not the mechanism (it would keep passing even if the real class
    were later renamed). test_metadata.py's twin stub covers the metadata
    builder directly; this one only needs a state_dict this callback can
    save and load back.
    """

    def __init__(self, hr_input_size: int):
        super().__init__()
        self.hparams = {"hr_input_size": hr_input_size}
        self.net = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)


def build_module_with_component() -> SRLightning:
    """An SRLightning carrying an extra, non-generator component under a
    `discriminator` attribute.

    Deliberately not SRGANLightning -- that class does not exist until a
    later PR, and this behaviour is about the callback, not the training
    loop. The component itself is a local stand-in (`_StubComponent`), not
    the real SRDiscriminator either -- that architecture is a later task's
    job; this only exercises the generic attribute/metadata mechanism.
    """
    module = build_module()
    module.discriminator = _StubComponent(hr_input_size=96)
    return module


def test_weights_checkpoint_can_save_a_named_component(tmp_path):
    module = build_module_with_component()
    # MagicMock, not Mock: _save_checkpoint's post-save logger notification
    # does `for logger in trainer.loggers`, and a plain Mock's auto-attrs
    # don't support __iter__ (raises TypeError) -- MagicMock's do.
    trainer = MagicMock(lightning_module=module, global_step=7, current_epoch=0)
    cb = SRWeightsCheckpoint(
        monitor_metric=None, keep_last=1, attribute="discriminator", dirpath=str(tmp_path)
    )
    cb.current_score = None

    cb._save_checkpoint(trainer, str(tmp_path / "d-weights-7.pt"))

    saved = torch.load(tmp_path / "d-weights-7.pt", weights_only=True)
    assert set(saved["state_dict"]) == set(module.discriminator.state_dict())
    # build_module()'s generator is SRCNN, whose top-level submodules are
    # feat/mapping/recon -- a "feat."-prefixed key here would mean the
    # generator's weights leaked into what must be a discriminator-only file.
    assert not any(key.startswith("feat.") for key in saved["state_dict"]), (
        "generator weights must not appear in a discriminator file"
    )
    assert saved["meta"]["kind"] == "component"


def test_two_weights_checkpoints_can_watch_different_components():
    """Generator and discriminator side by side is the whole reason ``attribute``
    exists, and it is unconstructable without this.

    Lightning refuses two stateful callbacks sharing a ``state_key``, and
    ``ModelCheckpoint``'s key is built from ``monitor``/``mode``/the cadence
    fields only — not from ``attribute``, ``dirpath`` or ``filename``. Two
    rolling weights callbacks differing only in which network they save
    therefore collide at ``Trainer`` construction, which is where the shipped
    SRGAN template puts them.
    """
    generator = SRWeightsCheckpoint(monitor_metric=None, attribute="model")
    discriminator = SRWeightsCheckpoint(monitor_metric=None, attribute="discriminator")

    assert generator.state_key != discriminator.state_key
    # Same config still keys the same, or saved callback state stops round-tripping.
    assert generator.state_key == SRWeightsCheckpoint(monitor_metric=None).state_key
    # The real failure path: Trainer validates the callback list on construction.
    lightning.Trainer(callbacks=[generator, discriminator], logger=False)


def _make_srcnn_datamodule(image_dir: Path):
    """Tiny SRDataModule (3 fixture images) mirroring test_integration.py's helper —
    real Trainer.fit needs a real datamodule to reach ModelCheckpoint's save path."""
    from sisr.training import SRDataModule

    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "use_tqdm": False,
            "cache_dir": str(image_dir / ".lmdb_cache_weights_ckpt_test"),
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(image_dir), "scale": 2},
    }
    return SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_sr_weights_checkpoint_writes_bare_payload_via_real_fit(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """End-to-end proof that Lightning's real save path invokes our
    ``_save_checkpoint`` override, that ``SRCheckpoint``/``SRWeightsCheckpoint``
    coexist in one ``dirpath`` without deleting each other's files, and that the
    bare ``.pt`` payload is exactly what the spec requires: no optimizer state,
    no ``model.``-prefixed keys, and the state_dict loads strict into a fresh
    bare model.

    This is the loud-failure guard for the private ``_save_checkpoint`` hook: if
    a future Lightning release stops routing saves through it, the ``.pt`` file
    would instead contain a full Lightning checkpoint dict (``optimizer_states``,
    ``callbacks``, ``model.``-prefixed ``state_dict``, ...) and every assertion
    below would fail.
    """
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4, momentum=0.9),
    )
    datamodule = _make_srcnn_datamodule(tiny_rgb_image_dir)
    # A dedicated subdir, not tmp_path itself: tiny_rgb_image_dir *is* tmp_path (the
    # fixture writes fixture PNGs directly into it), and ModelCheckpoint.setup warns
    # (-> error, under this suite's strict filter) if dirpath is non-empty at startup.
    ckpt_dir = tmp_path / "checkpoints"

    sr_ckpt = SRCheckpoint(monitor_metric="psnr/val/RGB", save_top_k=1, dirpath=str(ckpt_dir))
    sr_weights_ckpt = SRWeightsCheckpoint(
        monitor_metric="psnr/val/RGB", save_top_k=1, dirpath=str(ckpt_dir)
    )
    trainer = lightning.Trainer(
        max_epochs=-1,
        max_steps=3,
        val_check_interval=1,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[sr_ckpt, sr_weights_ckpt],
    )

    trainer.fit(module, datamodule=datamodule)

    ckpt_files = list(ckpt_dir.glob("sr-*.ckpt"))
    pt_files = list(ckpt_dir.glob("sr-weights-*.pt"))
    all_files = list(ckpt_dir.iterdir())
    # save_top_k=1 on each: exactly one file per callback survives repeated
    # val-triggered saves, and coexisting in one dirpath produced no cross-deletion.
    assert len(ckpt_files) == 1
    assert len(pt_files) == 1
    assert len(all_files) == 2

    full_checkpoint = torch.load(ckpt_files[0], weights_only=True)
    assert "optimizer_states" in full_checkpoint
    assert "sisr_meta" in full_checkpoint  # on_save_checkpoint fired here too
    assert all(k.startswith("model.") for k in full_checkpoint["state_dict"])

    bare = torch.load(pt_files[0], weights_only=True)
    assert set(bare.keys()) == {"state_dict", "meta"}
    assert "optimizer_states" not in bare
    assert not any(k.startswith("model.") for k in bare["state_dict"])
    assert bare["meta"]["format"] == "sisr-meta-v1"
    assert bare["meta"]["training"]["monitor"] == "psnr/val/RGB"

    fresh_model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    fresh_model.load_state_dict(bare["state_dict"], strict=True)  # the real proof


# ---------------------------------------------------------------------------
# Rolling last-N checkpoint mode (monitor_metric=None)
# ---------------------------------------------------------------------------


def _run_tiny_fit(
    callbacks: list, n_batches: int, ckpt_path: str | None = None
) -> lightning.Trainer:
    """Real `Trainer.fit` for `n_batches` steps, driving `callbacks`'s real save path.

    Rolling-mode retention depends on Lightning's actual save cadence, filename
    formatting, and `_remove_checkpoint` bookkeeping, so (unlike most of this
    file) a mocked hook can't stand in for the Trainer here. Images live in
    their own throwaway dir rather than a `tmp_path` fixture, since callers
    pass their checkpoint `dirpath` as `tmp_path` itself and
    `ModelCheckpoint.setup` errors (-> strict filterwarnings) if that dir is
    non-empty at startup. Teardown uses `ignore_errors=True`: LMDB keeps
    `data.mdb` memory-mapped past the dataset object's own lifetime on
    Windows, so a strict rmtree here intermittently raises `PermissionError`
    -- immaterial to what this helper actually verifies.
    """
    image_dir = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(seed=0)
        for i in range(3):
            arr = rng.integers(0, 256, size=(36, 36, 3), dtype=np.uint8)
            Image.fromarray(arr).save(image_dir / f"img_{i:02d}.png")

        model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
        module = SRLightning(
            model=model,
            processor=RGBProcessor(),
            eval_config=SREvalConfig(crop_border=0),
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4, momentum=0.9),
        )
        trainer = lightning.Trainer(
            max_epochs=-1,
            max_steps=n_batches,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=callbacks,
        )
        trainer.fit(module, datamodule=_make_srcnn_datamodule(image_dir), ckpt_path=ckpt_path)
        return trainer
    finally:
        shutil.rmtree(image_dir, ignore_errors=True)


@_ignore_gpu_warning
def test_rolling_mode_keeps_only_the_last_n(tmp_path):
    """monitor_metric=None keeps a sliding window of recent checkpoints.

    Needed because an adversarial objective makes PSNR/SSIM worse by design, so
    a metric-monitored save_top_k selects the LEAST adversarial model — very
    likely one from the first few thousand steps.
    """
    cb = SRCheckpoint(
        monitor_metric=None, keep_last=3, every_n_train_steps=2, dirpath=str(tmp_path)
    )
    _run_tiny_fit(callbacks=[cb], n_batches=20)
    files = sorted(p.name for p in tmp_path.glob("*.ckpt"))
    assert len(files) == 3, files
    # Not just any 3 -- the 3 *newest* by step (oldest-first deletion), matching
    # the mechanism spike's own result (20 batches, every_n_train_steps=2, keep_last=3
    # -> steps 16/18/20 survive out of the 10 saved at steps 2, 4, ..., 20).
    assert files == ["sr-16.ckpt", "sr-18.ckpt", "sr-20.ckpt"], files


@_ignore_gpu_warning
def test_rolling_mode_needs_no_monitor_validation(tmp_path: Path):
    """Rolling mode monitors nothing, so monitor validation must not fire.

    A bare `unittest.mock.Mock` trainer can't stand in for `.setup()` here —
    `ModelCheckpoint.setup` unconditionally calls
    `trainer.strategy.broadcast(dirpath)`, which returns a nonsense Mock
    instead of the path and crashes downstream filesystem resolution,
    regardless of monitor. `_make_bare_trainer()` (used by every other
    `.setup()` test in this file) is a real, unfitted Trainer that clears
    that machinery without running an actual loop, isolating this test to
    what it actually claims: that `_validate_monitor` doesn't fire.
    """
    cb = SRCheckpoint(monitor_metric=None, keep_last=2, dirpath=str(tmp_path))
    cb.setup(trainer=_make_bare_trainer(), pl_module=build_module(), stage="fit")  # must not raise


def test_srcheckpoint_rolling_filename_has_no_metric_placeholder(tmp_path: Path):
    """Rolling mode has no monitored metric, so the filename must be
    step-only — MEASURED FACT 3: omitting {step} would make every save
    overwrite the last."""
    ckpt = SRCheckpoint(monitor_metric=None, dirpath=str(tmp_path))
    assert ckpt.filename == "sr-{step}"
    assert ckpt.save_top_k == -1
    assert ckpt.monitor is None
    assert ckpt.keep_last == 3


@_ignore_gpu_warning
def test_sr_weights_checkpoint_rolling_mode_keeps_only_the_last_n(tmp_path):
    """SRWeightsCheckpoint's own _save_checkpoint override (bare .pt weights)
    must still compose with rolling deletion — it doesn't inherit
    ModelCheckpoint's save path the way SRCheckpoint does, so this is the
    place a mixin-only implementation would silently do nothing."""
    cb = SRWeightsCheckpoint(
        monitor_metric=None, keep_last=2, every_n_train_steps=2, dirpath=str(tmp_path)
    )
    _run_tiny_fit(callbacks=[cb], n_batches=20)
    files = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert files == ["sr-weights-18.pt", "sr-weights-20.pt"], files


def test_sr_weights_checkpoint_rolling_filename_has_no_metric_placeholder(tmp_path: Path):
    ckpt = SRWeightsCheckpoint(monitor_metric=None, dirpath=str(tmp_path))
    assert ckpt.filename == "sr-weights-{step}"
    assert ckpt.save_top_k == -1
    assert ckpt.monitor is None
    assert ckpt.keep_last == 3


@_ignore_gpu_warning
def test_rolling_checkpoints_coexist_without_cross_deletion(tmp_path):
    """SRCheckpoint and SRWeightsCheckpoint, both rolling, sharing one dirpath,
    must never delete each other's files — each tracks its own _rolling list
    of its own filepaths, and FILE_EXTENSION/filename_prefix already keep the
    two from colliding by name."""
    sr_ckpt = SRCheckpoint(
        monitor_metric=None, keep_last=2, every_n_train_steps=2, dirpath=str(tmp_path)
    )
    sr_weights_ckpt = SRWeightsCheckpoint(
        monitor_metric=None, keep_last=2, every_n_train_steps=2, dirpath=str(tmp_path)
    )
    _run_tiny_fit(callbacks=[sr_ckpt, sr_weights_ckpt], n_batches=20)
    ckpt_files = sorted(p.name for p in tmp_path.glob("sr-*.ckpt"))
    pt_files = sorted(p.name for p in tmp_path.glob("sr-weights-*.pt"))
    all_files = sorted(p.name for p in tmp_path.iterdir())
    assert ckpt_files == ["sr-18.ckpt", "sr-20.ckpt"], ckpt_files
    assert pt_files == ["sr-weights-18.pt", "sr-weights-20.pt"], pt_files
    assert len(all_files) == 4, all_files


@pytest.mark.parametrize("cls", [SRCheckpoint, SRWeightsCheckpoint])
def test_rolling_mode_rejects_keep_last_below_one(cls, tmp_path: Path):
    with pytest.raises(ValueError, match="keep_last"):
        cls(monitor_metric=None, keep_last=0, dirpath=str(tmp_path))


@pytest.mark.parametrize("cls", [SRCheckpoint, SRWeightsCheckpoint])
def test_enforce_rolling_window_is_noop_when_monitor_is_set(cls, tmp_path: Path):
    """With a monitor set, _enforce_rolling_window must not track saves or
    delete anything -- Lightning's own top-k selection owns retention then.

    A version that tracked saves unconditionally would eventually evict a
    file top-k still wants to keep once more than keep_last saves accumulate
    (e.g. save_top_k=1 keeps exactly 1 file at a time; FIFO-evicting by save
    count rather than by top-k's own bookkeeping can target that surviving
    file once the metric plateaus for keep_last+ saves), silently leaving the
    run with zero checkpoints until the next save recreates one. Calling
    _enforce_rolling_window directly, rather than through a full Trainer.fit,
    isolates this to the guard itself rather than needing a metric-plateau
    scenario reproduced end-to-end.
    """
    ckpt = cls(monitor_metric="psnr/val/RGB", save_top_k=1, dirpath=str(tmp_path))
    trainer = MagicMock()
    for i in range(ckpt.keep_last + 2):
        ckpt._enforce_rolling_window(trainer, f"fake-{i}{ckpt.FILE_EXTENSION}")
    assert ckpt._rolling == []
    trainer.strategy.remove_checkpoint.assert_not_called()


# ---------------------------------------------------------------------------
# Rolling window seeding on resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "prefix"), [(SRCheckpoint, "sr"), (SRWeightsCheckpoint, "sr-weights")]
)
def test_rolling_window_is_seeded_from_disk_on_resume(cls, prefix, tmp_path: Path):
    """A resumed run must adopt its own pre-resume files, not orphan them.

    Nothing persists the window (``ModelCheckpoint.state_dict`` carries
    ``monitor``/``best_*``/``kth_*``/``dirpath``/``last_model_path`` only), so
    without seeding every file written before the resume is never counted and
    never deleted — up to ``keep_last`` leaked per callback per resume, and the
    shipped adversarial template runs three rolling callbacks over a run long
    enough to be resumed repeatedly.

    ``dirpath`` is passed explicitly so the hook can be exercised without
    ``setup``'s dirpath resolution (which also warns on a non-empty directory,
    an error under this suite's strict filter).
    """
    for step in (4, 20, 12):
        (tmp_path / f"{prefix}-{step}{cls.FILE_EXTENSION}").touch()
    cb = cls(monitor_metric=None, keep_last=2, dirpath=str(tmp_path))

    cb.on_train_start(MagicMock(ckpt_path=str(tmp_path / "resumed-from.ckpt")), MagicMock())

    # Ordered by step, not lexicographically -- "12" sorts before "4" as text,
    # and deletion is oldest-first, so a string sort would evict the newest file.
    assert cb._rolling == [
        str(tmp_path / f"{prefix}-{step}{cls.FILE_EXTENSION}") for step in (4, 12, 20)
    ]


def test_rolling_seed_only_adopts_this_callbacks_own_files(tmp_path: Path):
    """Seeding must not let one rolling callback delete another's files.

    The shipped adversarial template puts three rolling callbacks in one
    ``dirpath`` (``sr-*.ckpt``, ``sr-weights-*.pt``, ``d-weights-*.pt``), and
    ``sr-weights`` is a strict prefix extension of ``sr`` — a prefix-only or
    extension-only filter adopts a sibling's files and eventually deletes them.
    """
    for name in ("sr-weights-2.pt", "d-weights-2.pt", "sr-2.ckpt", "last.ckpt", "sr-weights-x.pt"):
        (tmp_path / name).touch()
    cb = SRWeightsCheckpoint(monitor_metric=None, keep_last=3, dirpath=str(tmp_path))

    cb.on_train_start(MagicMock(ckpt_path=str(tmp_path / "resumed-from.ckpt")), MagicMock())

    assert cb._rolling == [str(tmp_path / "sr-weights-2.pt")]


def test_rolling_window_is_not_seeded_on_a_fresh_run(tmp_path: Path):
    """A fresh run must leave a populated dirpath alone.

    Seeding unconditionally would make a from-scratch run adopt — and, on its
    third save, silently delete — checkpoints a previous run wrote.
    """
    (tmp_path / "sr-weights-2.pt").touch()
    cb = SRWeightsCheckpoint(monitor_metric=None, keep_last=2, dirpath=str(tmp_path))

    cb.on_train_start(MagicMock(ckpt_path=None), MagicMock())

    assert cb._rolling == []


def test_rolling_seed_is_noop_when_monitor_is_set(tmp_path: Path):
    """With a monitor, Lightning's top-k owns retention — seeding would evict
    files top-k still wants, exactly as ``_enforce_rolling_window`` must not."""
    (tmp_path / "sr-weights-2.pt").touch()
    cb = SRWeightsCheckpoint(monitor_metric="psnr/val/RGB", save_top_k=1, dirpath=str(tmp_path))

    cb.on_train_start(MagicMock(ckpt_path=str(tmp_path / "resumed-from.ckpt")), MagicMock())

    assert cb._rolling == []


def test_seeded_window_deletes_the_oldest_pre_resume_file(tmp_path: Path):
    """Seeding is only worth anything if the next save then evicts a pre-resume
    file — the orphaning this fixes is a deletion that never happens."""
    for step in (2, 4):
        (tmp_path / f"sr-weights-{step}.pt").touch()
    cb = SRWeightsCheckpoint(monitor_metric=None, keep_last=2, dirpath=str(tmp_path))
    trainer = MagicMock(ckpt_path=str(tmp_path / "resumed-from.ckpt"))

    cb.on_train_start(trainer, MagicMock())
    cb._enforce_rolling_window(trainer, str(tmp_path / "sr-weights-6.pt"))

    trainer.strategy.remove_checkpoint.assert_called_once_with(str(tmp_path / "sr-weights-2.pt"))


@_ignore_gpu_warning
@pytest.mark.filterwarnings("ignore:Checkpoint directory .* exists and is not empty.:UserWarning")
def test_real_resume_keeps_the_window_at_keep_last(tmp_path: Path):
    """End-to-end: the hook fires with ``trainer.ckpt_path`` already assigned.

    The seeding gate reads ``trainer.ckpt_path``, which the checkpoint connector
    does not assign until it restores state — after every ``setup`` hook, which
    is why seeding lives in ``on_train_start``. Only a real resume proves the
    ordering; the hook-level tests above cannot.

    First fit saves at steps 2/4/6 and keeps 4/6. Resuming to step 12 saves at
    8/10/12. With the window seeded, four deletions leave exactly ``keep_last``
    files; unseeded, steps 4 and 6 are orphaned and six files survive.
    """
    ckpt_cb = SRCheckpoint(
        monitor_metric=None, keep_last=2, every_n_train_steps=2, dirpath=str(tmp_path)
    )
    _run_tiny_fit(callbacks=[ckpt_cb], n_batches=6)
    assert sorted(p.name for p in tmp_path.glob("*.ckpt")) == ["sr-4.ckpt", "sr-6.ckpt"]

    resumed_cb = SRCheckpoint(
        monitor_metric=None, keep_last=2, every_n_train_steps=2, dirpath=str(tmp_path)
    )
    _run_tiny_fit(callbacks=[resumed_cb], n_batches=12, ckpt_path=str(tmp_path / "sr-6.ckpt"))

    assert sorted(p.name for p in tmp_path.glob("*.ckpt")) == ["sr-10.ckpt", "sr-12.ckpt"]


# ---------------------------------------------------------------------------
# BenchmarkImageLogger validation/test hook bodies (mock-driven, no Trainer)
# ---------------------------------------------------------------------------


def _make_real_pl_module() -> SRLightning:
    """Real SRLightning with a small SRCNN — needed because the callback
    calls `pl_module(lr_img)` and `pl_module._build_psnr_tensors(sr, hr)`."""
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def _stub_dataset_with_img_paths(n: int, name: str) -> SimpleNamespace:
    """Stub Dataset exposing `img_paths` (used by callback for filename tags)."""
    paths = [Path(f"/synthetic/{name}_{i:03d}.png") for i in range(n)]
    return SimpleNamespace(img_paths=paths)


def _stub_dataloader(dataset) -> SimpleNamespace:
    return SimpleNamespace(dataset=dataset)


def test_benchmark_validation_batch_end_skips_primary_loader():
    """dataloader_idx=0 (primary val) must NOT populate the buffer."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"])
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = _make_real_pl_module()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    batch = (torch.rand(1, 3, 4, 4), torch.rand(1, 3, 4, 4))
    cb.on_validation_batch_end(
        trainer=SimpleNamespace(),
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )
    assert cb._buffer["Set5"] == []


def test_benchmark_validation_batch_end_collects_for_test_loader():
    """dataloader_idx >= 1 populates the buffer with a BenchmarkSample per image."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = _make_real_pl_module()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)

    ds = _stub_dataset_with_img_paths(n=2, name="Set5")
    trainer = SimpleNamespace(
        val_dataloaders=[
            _stub_dataloader(None),
            _stub_dataloader(ds),
        ],  # idx 0 = primary, idx 1 = Set5
    )
    batch = (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    assert len(cb._buffer["Set5"]) == 2
    sample = cb._buffer["Set5"][0]
    assert sample.filename == "Set5_000"
    assert "RGB" in sample.psnr
    assert "RGB" in sample.ssim and "Y" in sample.ssim  # eval_config.ssim_channels default
    assert isinstance(sample.ssim["RGB"], float)


def test_benchmark_collect_batch_buffers_no_image_tensors():
    """The buffer must hold BenchmarkSample(filename, psnr, ssim) only -- no
    LR/SR/HR tensors -- even when should_log_images is True, since image
    strips are now streamed directly from _collect_batch instead of being
    deferred to epoch end."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = _make_real_pl_module()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer = SimpleNamespace(val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)])
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    sample = cb._buffer["Set5"][0]
    assert isinstance(sample, BenchmarkSample)
    for value in sample:
        assert not isinstance(value, torch.Tensor)


def test_collect_batch_metric_values_match_pre_change_host_copy_first_ordering():
    """D3 equivalence proof: the pre-change code moved each image to CPU
    FIRST (``sr[i].unsqueeze(0).cpu()``), THEN cropped, THEN built metric
    tensors + PSNR/SSIM; the shipped _collect_batch crops the (on-device)
    per-image slice directly and computes PSNR/SSIM on it without any
    intervening host copy. Both orderings must agree exactly on every
    configured metric key. CPU-only (no GPU dependency), so this runs on CI.
    """
    torch.manual_seed(0)
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    pl_module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(
            crop_border=3,
            psnr_channels=["RGB", "YCbCr"],
            separate_psnr=True,
            ssim_channels=["RGB", "Y"],
        ),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    lr_img = torch.rand(4, 3, 20, 20)
    hr_img = torch.rand(4, 3, 20, 20)
    with torch.no_grad():
        sr, hr_cropped = pl_module.predict_rgb(lr_img, hr_img)

    n = pl_module.eval_config.crop_border
    psnr_keys = pl_module.eval_config.psnr_keys
    ssim_keys = pl_module.eval_config.ssim_keys
    assert n > 0 and len(psnr_keys) > 1 and len(ssim_keys) > 1  # actually exercises both

    for i in range(lr_img.size(0)):
        # Pre-change ordering: host-copy the per-image slice FIRST, then crop.
        sr_old = sr[i].unsqueeze(0).cpu()
        hr_old = hr_cropped[i].unsqueeze(0).cpu()
        sr_old = sr_old[..., n:-n, n:-n]
        hr_old = hr_old[..., n:-n, n:-n]
        metric_tensors_old = pl_module._build_metric_tensors(sr_old, hr_old)
        psnr_old = {
            key: torchmetrics.functional.image.peak_signal_noise_ratio(
                *metric_tensors_old[key], data_range=1.0
            ).item()
            for key in psnr_keys
        }
        ssim_old = {
            key: torchmetrics.functional.image.structural_similarity_index_measure(
                *metric_tensors_old[key], data_range=1.0
            ).item()
            for key in ssim_keys
        }

        # Shipped ordering: crop the per-image slice directly, no intervening copy.
        sr_new = sr[i : i + 1][..., n:-n, n:-n]
        hr_new = hr_cropped[i : i + 1][..., n:-n, n:-n]
        metric_tensors_new = pl_module._build_metric_tensors(sr_new, hr_new)
        psnr_new = {
            key: torchmetrics.functional.image.peak_signal_noise_ratio(
                *metric_tensors_new[key], data_range=1.0
            ).item()
            for key in psnr_keys
        }
        ssim_new = {
            key: torchmetrics.functional.image.structural_similarity_index_measure(
                *metric_tensors_new[key], data_range=1.0
            ).item()
            for key in ssim_keys
        }

        for key in psnr_keys:
            assert psnr_new[key] == pytest.approx(psnr_old[key], abs=1e-6), f"psnr[{key}]"
        for key in ssim_keys:
            assert ssim_new[key] == pytest.approx(ssim_old[key], abs=1e-6), f"ssim[{key}]"


def test_benchmark_logger_uses_module_ssim_impl():
    """BenchmarkImageLogger must not re-implement the metric: with
    ssim_impl='daala' its buffered SSIM has to equal sisr.ssim.daala_ssim, not
    torchmetrics. Guards against the two metric paths drifting apart."""
    import sisr.ssim

    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    pl_module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(crop_border=0, ssim_channels=["Y"], ssim_impl="daala"),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)

    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer = SimpleNamespace(val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)])
    lr_img = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    hr_img = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=(lr_img, hr_img),
        batch_idx=0,
        dataloader_idx=1,
    )

    with torch.no_grad():
        sr, hr_cropped = pl_module.predict_rgb(lr_img, hr_img)
    metric_tensors = pl_module._build_metric_tensors(sr, hr_cropped)
    expected = sisr.ssim.daala_ssim(*metric_tensors["Y"]).item()

    sample = cb._buffer["Set5"][0]
    assert sample.ssim["Y"] == pytest.approx(expected, rel=1e-9)


def test_benchmark_validation_epoch_end_logs_means():
    """on_validation_epoch_end consumes the buffer and emits per-dataset mean
    PSNR + SSIM via pl_module.log (verified by mocking the log method)."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=99)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = MagicMock()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    cb._buffer["Set5"] = [
        BenchmarkSample("img_0", {"RGB": 30.0}, {"RGB": 0.9}, {}),
        BenchmarkSample("img_1", {"RGB": 32.0}, {"RGB": 0.85}, {}),
    ]
    cb.on_validation_epoch_end(trainer=SimpleNamespace(), pl_module=pl_module)
    # Two log calls per dataset: psnr + ssim.
    log_keys = [call.args[0] for call in pl_module.log.call_args_list]
    assert "psnr/Set5/RGB" in log_keys
    assert "ssim/Set5/RGB" in log_keys
    # Mean PSNR = (30.0 + 32.0) / 2 = 31.0
    psnr_call = next(c for c in pl_module.log.call_args_list if c.args[0] == "psnr/Set5/RGB")
    assert psnr_call.args[1] == pytest.approx(31.0)
    # Mean SSIM = (0.9 + 0.85) / 2 = 0.875
    ssim_call = next(c for c in pl_module.log.call_args_list if c.args[0] == "ssim/Set5/RGB")
    assert ssim_call.args[1] == pytest.approx(0.875)


def test_benchmark_collect_batch_emits_add_image_and_add_scalar(tmp_path: Path, monkeypatch):
    """With log_per_image_metrics=True and on the log interval, _collect_batch
    must call experiment.add_image once (the bicubic|SR|HR strip) and
    experiment.add_scalar for the per-image psnr + ssim -- streamed immediately
    from the batch hook rather than deferred to epoch end."""
    import lightning.pytorch.loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    experiment = tb_logger.experiment  # materialize the SummaryWriter before wrapping
    add_image = MagicMock(wraps=experiment.add_image)
    add_scalar = MagicMock(wraps=experiment.add_scalar)
    monkeypatch.setattr(experiment, "add_image", add_image)
    monkeypatch.setattr(experiment, "add_scalar", add_scalar)

    cb = BenchmarkImageLogger(
        dataset_names=["Set5"], log_every_n_val_runs=1, log_per_image_metrics=True
    )
    trainer = SimpleNamespace(
        datamodule=None,
        loggers=[tb_logger],
        # Two axes, deliberately unequal: emission must follow the batch-counted
        # one, so a stub where they agree could not observe a regression here.
        global_step=42,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=21)),
    )
    cb.setup(trainer, pl_module=None, stage="fit")
    pl_module = _make_real_pl_module()  # SRCNN + RGBProcessor: LR/SR/HR all 16x16
    cb.on_validation_epoch_start(trainer=trainer, pl_module=pl_module)

    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer.val_dataloaders = [_stub_dataloader(None), _stub_dataloader(ds)]
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    tb_logger.finalize("success")

    add_image.assert_called_once()
    tag, strip = add_image.call_args.args[0], add_image.call_args.args[1]
    assert tag == "Set5/Set5_000"
    assert strip.ndim == 3 and strip.shape[0] == 3  # (C, H, W) triptych
    # psnr_channels default ['RGB'] + ssim_channels default ['RGB', 'Y'].
    scalar_tags = [c.args[0] for c in add_scalar.call_args_list]
    assert scalar_tags == [
        "per_image/Set5/psnr/RGB/Set5_000",
        "per_image/Set5/ssim/RGB/Set5_000",
        "per_image/Set5/ssim/Y/Set5_000",
    ]


def test_benchmark_collect_batch_default_omits_per_image_scalars_but_keeps_images(
    tmp_path: Path, monkeypatch
):
    """log_per_image_metrics defaults to False: image strips still stream
    (gated only by should_log_images), but no per_image/... scalar is
    emitted."""
    import lightning.pytorch.loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    experiment = tb_logger.experiment
    add_image = MagicMock(wraps=experiment.add_image)
    add_scalar = MagicMock(wraps=experiment.add_scalar)
    monkeypatch.setattr(experiment, "add_image", add_image)
    monkeypatch.setattr(experiment, "add_scalar", add_scalar)

    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    assert cb.log_per_image_metrics is False
    trainer = SimpleNamespace(
        datamodule=None,
        loggers=[tb_logger],
        # Two axes, deliberately unequal: emission must follow the batch-counted
        # one, so a stub where they agree could not observe a regression here.
        global_step=42,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=21)),
    )
    cb.setup(trainer, pl_module=None, stage="fit")
    pl_module = _make_real_pl_module()
    cb.on_validation_epoch_start(trainer=trainer, pl_module=pl_module)

    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer.val_dataloaders = [_stub_dataloader(None), _stub_dataloader(ds)]
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    tb_logger.finalize("success")

    add_image.assert_called_once()  # image strip unaffected by the flag
    add_scalar.assert_not_called()  # no per_image/... scalar emitted


def test_benchmark_collect_batch_log_per_image_metrics_decoupled_from_image_strips(
    tmp_path: Path, monkeypatch
):
    """log_per_image_metrics=True must emit per-image scalars even on a val run
    where should_log_images is False (log_every_n_val_runs throttles images,
    not per-image metrics) — proving the two concerns are independently gated."""
    import lightning.pytorch.loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    experiment = tb_logger.experiment
    add_image = MagicMock(wraps=experiment.add_image)
    add_scalar = MagicMock(wraps=experiment.add_scalar)
    monkeypatch.setattr(experiment, "add_image", add_image)
    monkeypatch.setattr(experiment, "add_scalar", add_scalar)

    cb = BenchmarkImageLogger(
        dataset_names=["Set5"], log_every_n_val_runs=99, log_per_image_metrics=True
    )
    trainer = SimpleNamespace(
        datamodule=None,
        loggers=[tb_logger],
        # Two axes, deliberately unequal: emission must follow the batch-counted
        # one, so a stub where they agree could not observe a regression here.
        global_step=42,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=21)),
    )
    cb.setup(trainer, pl_module=None, stage="fit")
    pl_module = _make_real_pl_module()
    cb.on_validation_epoch_start(trainer=trainer, pl_module=pl_module)
    assert cb._on_image_log_interval() is False  # confirms should_log_images will be False

    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer.val_dataloaders = [_stub_dataloader(None), _stub_dataloader(ds)]
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    tb_logger.finalize("success")

    add_image.assert_not_called()  # images still throttled
    scalar_tags = [c.args[0] for c in add_scalar.call_args_list]
    assert scalar_tags == [
        "per_image/Set5/psnr/RGB/Set5_000",
        "per_image/Set5/ssim/RGB/Set5_000",
        "per_image/Set5/ssim/Y/Set5_000",
    ]


def test_benchmark_collect_batch_image_strip_first_panel_is_bicubic_at_hr_size(
    tmp_path: Path, monkeypatch
):
    """The first triptych panel must be the bicubic-upsampled LR at HR size,
    not the original LR or a zero-padded LR. Locks the Bicubic|SR|HR layout."""
    import lightning.pytorch.loggers as pl_loggers
    import torchvision.utils

    captured: list[list[torch.Tensor]] = []
    real_make_grid = torchvision.utils.make_grid

    def spy_make_grid(tensors, *args, **kwargs):
        captured.append(list(tensors))
        return real_make_grid(tensors, *args, **kwargs)

    monkeypatch.setattr(torchvision.utils, "make_grid", spy_make_grid)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    trainer = SimpleNamespace(
        datamodule=None,
        loggers=[tb_logger],
        # Two axes, deliberately unequal: emission must follow the batch-counted
        # one, so a stub where they agree could not observe a regression here.
        global_step=42,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=21)),
    )
    cb.setup(trainer, pl_module=None, stage="fit")

    pl_module = _make_real_pl_module()
    # Fix predict_rgb's output so the SR/HR panel content is known exactly,
    # independent of the (untrained) model's actual forward pass.
    sr_fixed = torch.rand(1, 3, 16, 16)
    hr_cropped_fixed = torch.rand(1, 3, 16, 16)
    monkeypatch.setattr(
        pl_module, "predict_rgb", MagicMock(return_value=(sr_fixed, hr_cropped_fixed))
    )
    cb.on_validation_epoch_start(trainer=trainer, pl_module=pl_module)

    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer.val_dataloaders = [_stub_dataloader(None), _stub_dataloader(ds)]
    lr_batch = torch.rand(1, 3, 4, 4)
    hr_batch = torch.rand(1, 3, 16, 16)
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=(lr_batch, hr_batch),
        batch_idx=0,
        dataloader_idx=1,
    )
    tb_logger.finalize("success")

    assert len(captured) == 1
    panels = captured[0]
    assert len(panels) == 3
    expected_bicubic = BenchmarkImageLogger._bicubic_to(lr_batch[0], hr_batch[0].shape[-2:])
    torch.testing.assert_close(panels[0], expected_bicubic)
    # SR is already HR-sized so _pad_to_match is a no-op; HR panel is the
    # original (uncropped) batch HR, not predict_rgb's cropped hr_cropped.
    torch.testing.assert_close(panels[1], sr_fixed[0])
    torch.testing.assert_close(panels[2], hr_batch[0])


def test_benchmark_collect_batch_image_strip_upsamples_lr_to_hr_size(tmp_path: Path, monkeypatch):
    """For an upscaling model (SRResNet) LR (4x4) is smaller than SR/HR (16x16);
    the strip must bicubic-upsample LR to HR size, so add_image is called once
    with a panel taller than the LR (not crash make_grid on a size mismatch)."""
    import lightning.pytorch.loggers as pl_loggers

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    experiment = tb_logger.experiment
    add_image = MagicMock(wraps=experiment.add_image)
    monkeypatch.setattr(experiment, "add_image", add_image)

    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    trainer = SimpleNamespace(
        datamodule=None,
        loggers=[tb_logger],
        # Two axes, deliberately unequal: emission must follow the batch-counted
        # one, so a stub where they agree could not observe a regression here.
        global_step=42,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_batches_that_stepped=21)),
    )
    cb.setup(trainer, pl_module=None, stage="fit")

    pl_module = _make_real_pl_module()
    sr_fixed = torch.rand(1, 3, 16, 16)
    hr_cropped_fixed = torch.rand(1, 3, 16, 16)
    monkeypatch.setattr(
        pl_module, "predict_rgb", MagicMock(return_value=(sr_fixed, hr_cropped_fixed))
    )
    cb.on_validation_epoch_start(trainer=trainer, pl_module=pl_module)

    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer.val_dataloaders = [_stub_dataloader(None), _stub_dataloader(ds)]
    lr_batch = torch.rand(1, 3, 4, 4)  # LR 4x4, SR/HR (mocked) 16x16 -- x4 model
    hr_batch = torch.rand(1, 3, 16, 16)
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=(lr_batch, hr_batch),
        batch_idx=0,
        dataloader_idx=1,
    )  # must not raise
    tb_logger.finalize("success")

    add_image.assert_called_once()
    strip = add_image.call_args.args[1]
    # make_grid pads each panel by 2px per side -> HR-sized panels give height
    # 16 + 4 = 20; the point is it is upsampled to HR (>> the 4px LR).
    assert strip.shape[-2] == 20
    assert strip.shape[-2] > 4


def test_benchmark_test_batch_end_collects_with_zero_indexed_mapping():
    """During cli test, dataloader_idx 0..N-1 are all test loaders."""
    cb = BenchmarkImageLogger(dataset_names=["Set5", "Set14"])
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="test")
    pl_module = _make_real_pl_module()
    cb.on_test_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    ds = _stub_dataset_with_img_paths(n=1, name="Set14")
    trainer = SimpleNamespace(
        test_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)],
    )
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_test_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    assert len(cb._buffer["Set14"]) == 1
    assert len(cb._buffer["Set5"]) == 0


def test_benchmark_test_epoch_end_logs_means():
    cb = BenchmarkImageLogger(dataset_names=["Set5"])
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="test")
    pl_module = MagicMock()
    cb.on_test_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    cb._buffer["Set5"] = [BenchmarkSample("img_0", {"RGB": 28.0}, {"RGB": 0.7}, {})]
    cb.on_test_epoch_end(trainer=SimpleNamespace(), pl_module=pl_module)
    log_keys = [call.args[0] for call in pl_module.log.call_args_list]
    assert "psnr/Set5/RGB" in log_keys
    assert "ssim/Set5/RGB" in log_keys


# ---------------------------------------------------------------------------
# BenchmarkImageLogger perceptual metrics (LPIPS/DISTS) per benchmark set
# ---------------------------------------------------------------------------


def test_benchmark_collect_batch_populates_perceptual_dict(monkeypatch):
    """_collect_batch must pass the per-image, border-cropped SR/HR pair into
    pl_module._mean_perceptual (mocked here at its perceptual_score seam, so no
    real LPIPS/DISTS weights load) -- not the un-cropped batch-level tensors or
    anything else in scope. The mock's return depends on sr/hr
    (``sr.sum() - hr.sum()``) rather than being a constant, so a wrong tensor
    pair changes the recorded value instead of passing regardless of it; a
    non-zero crop_border also means a wrong (un-cropped) pair differs in shape,
    not just content."""
    monkeypatch.setattr(
        "sisr.training.lightning_module.perceptual_score",
        lambda name, sr, hr, lpips_net: sr.sum() - hr.sum(),
    )
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    pl_module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(crop_border=2, perceptual_metrics=["lpips"]),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    sr_fixed = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    hr_cropped_fixed = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    monkeypatch.setattr(
        pl_module, "predict_rgb", MagicMock(return_value=(sr_fixed, hr_cropped_fixed))
    )
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer = SimpleNamespace(val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)])
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )

    n = pl_module.eval_config.crop_border
    expected_sr = sr_fixed[0:1][..., n:-n, n:-n]
    expected_hr = hr_cropped_fixed[0:1][..., n:-n, n:-n]
    expected = (expected_sr.sum() - expected_hr.sum()).item()

    sample = cb._buffer["Set5"][0]
    assert sample.perceptual == pytest.approx({"lpips": expected})


def test_benchmark_logs_perceptual_per_set():
    """Test-set scoring must offer the same metric families validation does.

    _flush_buffer only reduces already-buffered floats -- perceptual scoring
    itself happens upstream in _collect_batch (see
    test_benchmark_collect_batch_populates_perceptual_dict) -- so a plain
    MagicMock pl_module is enough here, mirroring
    test_benchmark_validation_epoch_end_logs_means.
    """
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=99)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = MagicMock()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    cb._buffer["Set5"] = [
        BenchmarkSample("a.png", {"RGB": 30.0}, {"RGB": 0.9}, {"lpips": 0.5}),
        BenchmarkSample("b.png", {"RGB": 32.0}, {"RGB": 0.8}, {"lpips": 0.3}),
    ]
    cb.on_validation_epoch_end(trainer=SimpleNamespace(), pl_module=pl_module)

    lpips_call = next(c for c in pl_module.log.call_args_list if c.args[0] == "lpips/Set5")
    assert lpips_call.args[1] == pytest.approx(0.4)  # mean of 0.5 and 0.3


# ---------------------------------------------------------------------------
# WeightHistogramLogger on-cadence path
# ---------------------------------------------------------------------------


def test_weight_histogram_logger_calls_add_histogram_for_model_params_only(
    tmp_path: Path, monkeypatch
):
    """On-cadence, the logger must call experiment.add_histogram exactly once for
    the single `model.`-prefixed parameter (the non-model param is filtered out),
    with the prefix-grouped tag. The old test only checked it didn't raise, so a
    broken `model.` filter or a missing emit would have passed."""
    import lightning.pytorch.loggers as pl_loggers

    cb = WeightHistogramLogger(log_every_n_steps=1)
    pl_module = MagicMock()
    pl_module.named_parameters = MagicMock(
        return_value=[
            ("model.feat.0.weight", torch.nn.Parameter(torch.rand(4, 3, 3, 3))),
            ("non_model_param", torch.nn.Parameter(torch.rand(4))),  # filtered out
        ]
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    experiment = tb_logger.experiment
    add_histogram = MagicMock(wraps=experiment.add_histogram)
    monkeypatch.setattr(experiment, "add_histogram", add_histogram)

    trainer = _make_step_axis_trainer(global_step=2, batches_that_stepped=1, loggers=[tb_logger])
    cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
    tb_logger.finalize("success")

    add_histogram.assert_called_once()
    # name.split('.', 2) -> ["model","feat","0.weight"]; tag = "model" + "." + "feat/0.weight"
    assert add_histogram.call_args.args[0] == "model.feat/0.weight"


# ---------------------------------------------------------------------------
# BenchmarkImageLogger crop_border migration to eval_config
# ---------------------------------------------------------------------------


def test_crop_border_init_arg_rejected():
    """crop_border was dropped from BenchmarkImageLogger; now lives on eval_config."""
    with pytest.raises(TypeError):
        BenchmarkImageLogger(crop_border=3)


def test_benchmark_collect_batch_crops_per_eval_config():
    """When eval_config.crop_border=3, _collect_batch crops 3 pixels per edge
    before computing per-image PSNR/SSIM."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    # Real SRLightning with crop_border=3 in eval_config.
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    pl_module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(crop_border=3),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer = SimpleNamespace(
        val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)],
    )
    # 16x16 input; after crop_border=3 sides, the inner 10x10 region drives PSNR/SSIM.
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    assert len(cb._buffer["Set5"]) == 1
    sample = cb._buffer["Set5"][0]
    assert "RGB" in sample.psnr
    assert "RGB" in sample.ssim
    assert isinstance(sample.ssim["RGB"], float)


def test_benchmark_collect_batch_routes_through_processor():
    """SRCNN+YChannelProcessor: _collect_batch must extract Y from RGB LR before the
    model forward, then reconstruct SR back to RGB. Bypassing the processor would
    feed 3-channel RGB to a 1-channel Conv2d and crash with a shape mismatch (proven
    here by the forward completing and yielding the expected metric key sets)."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    # 1-channel SRCNN paired with YChannelProcessor — production SRCNN template shape.
    model = SRCNN(num_channels=1, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    pl_module = SRLightning(
        model=model,
        processor=YChannelProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(crop_border=0, psnr_channels=["RGB", "YCbCr"]),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer = SimpleNamespace(
        val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)],
    )
    # Batch is 3-channel RGB (dataset-format); the processor must extract Y before the model.
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    assert len(cb._buffer["Set5"]) == 1
    sample = cb._buffer["Set5"][0]
    assert "RGB" in sample.psnr
    assert "YCbCr" in sample.psnr
    assert "RGB" in sample.ssim and "Y" in sample.ssim  # eval_config.ssim_channels default
    assert isinstance(sample.ssim["RGB"], float)


def test_benchmark_collect_batch_psnr_dict_matches_configured_keys_separate_false():
    """Regression: a real run's metrics.csv contained 8 PSNR keys
    (RGB/R/G/B/YCbCr/Y/Cb/Cr) for a config requesting only ['RGB', 'YCbCr']
    with separate_psnr=False — because _collect_batch iterated every tensor
    `_build_psnr_tensors` populates for the colorspace *family*, not the
    configured key set. The emitted key set must equal `eval_config.psnr_keys`
    exactly."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    pl_module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(
            crop_border=0, psnr_channels=["RGB", "YCbCr"], separate_psnr=False
        ),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer = SimpleNamespace(val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)])
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    sample = cb._buffer["Set5"][0]
    assert set(sample.psnr.keys()) == set(pl_module.eval_config.psnr_keys) == {"RGB", "YCbCr"}


def test_benchmark_collect_batch_psnr_dict_matches_configured_keys_separate_true():
    """separate_psnr=True must surface per-channel keys alongside the
    aggregate key for the requested colorspace only — no keys from an
    unrequested colorspace family leak in."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    pl_module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(crop_border=0, psnr_channels=["RGB"], separate_psnr=True),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    ds = _stub_dataset_with_img_paths(n=1, name="Set5")
    trainer = SimpleNamespace(val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)])
    batch = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    sample = cb._buffer["Set5"][0]
    assert set(sample.psnr.keys()) == set(pl_module.eval_config.psnr_keys) == {"R", "G", "B", "RGB"}


# ---------------------------------------------------------------------------
# BenchmarkImageLogger consumes the public predict_rgb seam + SRDataset
# ---------------------------------------------------------------------------


def test_benchmark_collect_batch_routes_through_predict_rgb():
    """_collect_batch must call the public pl_module.predict_rgb -- proving no
    re-implemented, divergent forward path remains in the callback."""
    pl_module = _make_real_pl_module()  # SRCNN + RGBProcessor, crop_border=0
    batch = (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16))

    spy = MagicMock(wraps=pl_module.predict_rgb)
    pl_module.predict_rgb = spy

    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    ds = _stub_dataset_with_img_paths(n=2, name="Set5")
    trainer = SimpleNamespace(
        val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)],
    )
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )

    spy.assert_called_once()
    call_args, _ = spy.call_args
    torch.testing.assert_close(call_args[0], batch[0])
    torch.testing.assert_close(call_args[1], batch[1])
    assert len(cb._buffer["Set5"]) == 2


def test_prediction_writer_creates_output_dir(tmp_path: Path):
    out_dir = tmp_path / "preds" / "nested"
    SRPredictionWriter(output_dir=out_dir)
    assert out_dir.is_dir()


def test_prediction_writer_writes_png_named_after_input(tmp_path: Path):
    from PIL import Image

    out_dir = tmp_path / "preds"
    writer = SRPredictionWriter(output_dir=out_dir)

    ds = _stub_dataset_with_img_paths(n=2, name="lr")
    trainer = SimpleNamespace(predict_dataloaders=_stub_dataloader(ds))
    prediction = torch.rand(2, 3, 8, 8)
    writer.write_on_batch_end(
        trainer=trainer,
        pl_module=None,
        prediction=prediction,
        batch_indices=[0, 1],
        batch=None,
        batch_idx=0,
        dataloader_idx=0,
    )
    written = sorted(p.name for p in out_dir.glob("*.png"))
    assert written == ["lr_000.png", "lr_001.png"]
    with Image.open(out_dir / "lr_000.png") as img:
        assert img.size == (8, 8)


def test_prediction_writer_handles_dataloader_list():
    """trainer.predict_dataloaders may be a list (Lightning's CombinedLoader
    behaviour for multi-loader predict); the writer must index it by
    dataloader_idx rather than assuming a single bare DataLoader."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "preds"
        writer = SRPredictionWriter(output_dir=out_dir)

        ds = _stub_dataset_with_img_paths(n=1, name="lr")
        trainer = SimpleNamespace(predict_dataloaders=[_stub_dataloader(ds)])
        prediction = torch.rand(1, 3, 4, 4)
        writer.write_on_batch_end(
            trainer=trainer,
            pl_module=None,
            prediction=prediction,
            batch_indices=[0],
            batch=None,
            batch_idx=0,
            dataloader_idx=0,
        )
        assert (out_dir / "lr_000.png").exists()


def test_prediction_writer_clamps_out_of_range_values(tmp_path: Path):
    """Model output isn't guaranteed to land in [0, 1]; the writer must clamp
    before saving rather than let save_image wrap/overflow."""
    out_dir = tmp_path / "preds"
    writer = SRPredictionWriter(output_dir=out_dir)

    ds = _stub_dataset_with_img_paths(n=1, name="lr")
    trainer = SimpleNamespace(predict_dataloaders=_stub_dataloader(ds))
    prediction = torch.full((1, 3, 4, 4), 1.5)  # out of [0, 1]
    writer.write_on_batch_end(
        trainer=trainer,
        pl_module=None,
        prediction=prediction,
        batch_indices=[0],
        batch=None,
        batch_idx=0,
        dataloader_idx=0,
    )
    import numpy as np
    from PIL import Image

    with Image.open(out_dir / "lr_000.png") as img:
        arr = np.array(img)
    assert (arr == 255).all()


def test_benchmark_collect_batch_consumes_srdataset_img_paths(tiny_rgb_image_dir: Path):
    """The callback resolves filenames from a real SRDataset's declared
    .img_paths contract (not a duck-typed attr)."""
    from sisr.datasets.base import SRDataset
    from sisr.datasets.srresnet import ValidationDataset

    ds = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=2)
    assert isinstance(ds, SRDataset)

    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = _make_real_pl_module()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    trainer = SimpleNamespace(
        val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)],
    )
    batch = (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16))
    cb.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=1,
    )
    fnames = [entry[0] for entry in cb._buffer["Set5"]]
    assert fnames == [p.stem for p in ds.img_paths[:2]]  # e.g. ["img_00", "img_01"]
