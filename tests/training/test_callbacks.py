import functools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import lightning
import pytest
import torch
import torchmetrics.functional
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.exceptions import MisconfigurationException

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
    trainer = SimpleNamespace(global_step=10)
    pl_module = _make_gradnorm_pl_module()
    cb.on_after_backward(trainer, pl_module)
    pl_module.log.assert_called_once()
    args, _ = pl_module.log.call_args
    assert args[0] == "diag/grad_norm"
    assert args[1] == pytest.approx(10.0)


def test_grad_norm_logger_skips_off_cadence():
    cb = GradNormLogger(log_every_n_steps=10)
    trainer = SimpleNamespace(global_step=7)
    pl_module = _make_gradnorm_pl_module()
    cb.on_after_backward(trainer, pl_module)
    pl_module.log.assert_not_called()


def test_grad_norm_logger_handles_none_grads():
    cb = GradNormLogger(log_every_n_steps=1)
    pl_module = MagicMock()
    p = torch.zeros(4, requires_grad=True)
    p.grad = None
    pl_module.parameters = MagicMock(return_value=[p])
    trainer = SimpleNamespace(global_step=1)
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
    trainer = SimpleNamespace(global_step=1)
    pl_module = _make_gradnorm_pl_module()
    cb.on_after_backward(trainer, pl_module)
    args, _ = pl_module.log.call_args
    assert args[0] == "diag/grad_norm"
    assert args[1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# WeightHistogramLogger
# ---------------------------------------------------------------------------


def test_weight_histogram_logger_skips_off_cadence():
    cb = WeightHistogramLogger(log_every_n_steps=10)
    trainer = SimpleNamespace(global_step=7, loggers=[])
    pl_module = MagicMock()
    cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
    pl_module.named_parameters.assert_not_called()


def test_weight_histogram_logger_skips_when_no_tb_logger():
    cb = WeightHistogramLogger(log_every_n_steps=1)
    trainer = SimpleNamespace(global_step=1, loggers=[])
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


def test_sr_weights_checkpoint_save_checkpoint_is_actually_overridden():
    """Cheap static guard: the override must exist as a distinct method, not
    silently inherit ModelCheckpoint's (which would write full, optimizer-bearing
    checkpoints under the .pt extension). The real, functional guard against a
    Lightning release routing saves through a different private method entirely
    is test_sr_weights_checkpoint_writes_bare_payload_via_real_fit below, which
    exercises the actual save path end-to-end."""
    assert SRWeightsCheckpoint._save_checkpoint is not ModelCheckpoint._save_checkpoint


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
        BenchmarkSample("img_0", {"RGB": 30.0}, {"RGB": 0.9}),
        BenchmarkSample("img_1", {"RGB": 32.0}, {"RGB": 0.85}),
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
    trainer = SimpleNamespace(datamodule=None, loggers=[tb_logger], global_step=42)
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
    trainer = SimpleNamespace(datamodule=None, loggers=[tb_logger], global_step=42)
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
    trainer = SimpleNamespace(datamodule=None, loggers=[tb_logger], global_step=42)
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
    trainer = SimpleNamespace(datamodule=None, loggers=[tb_logger], global_step=42)
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
    trainer = SimpleNamespace(datamodule=None, loggers=[tb_logger], global_step=42)
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
    cb._buffer["Set5"] = [BenchmarkSample("img_0", {"RGB": 28.0}, {"RGB": 0.7})]
    cb.on_test_epoch_end(trainer=SimpleNamespace(), pl_module=pl_module)
    log_keys = [call.args[0] for call in pl_module.log.call_args_list]
    assert "psnr/Set5/RGB" in log_keys
    assert "ssim/Set5/RGB" in log_keys


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

    trainer = SimpleNamespace(global_step=1, loggers=[tb_logger])
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
