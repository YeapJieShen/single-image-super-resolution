import functools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sisr.models.srcnn import SRCNN
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import (
    BenchmarkImageLogger,
    GradNormLogger,
    SRCheckpoint,
    SREvalConfig,
    SRLightning,
    SRTrainingConfig,
    WeightHistogramLogger,
)


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
    expected = torch.nn.functional.interpolate(
        lr.unsqueeze(0), size=(16, 16), mode="bicubic", align_corners=False
    ).squeeze(0).clamp(0.0, 1.0)
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
    assert args[0] == "grad_norm"
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
    """dataloader_idx >= 1 populates the buffer with metrics + (since
    log_every_n_val_runs=1) cached LR/SR/HR tensors."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = _make_real_pl_module()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)

    ds = _stub_dataset_with_img_paths(n=2, name="Set5")
    trainer = SimpleNamespace(
        val_dataloaders=[_stub_dataloader(None), _stub_dataloader(ds)],  # idx 0 = primary, idx 1 = Set5
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
    # Each entry: (filename, lr|None, sr|None, hr|None, psnr_dict, ssim).
    fname, lr, sr, hr, psnr, ssim = cb._buffer["Set5"][0]
    assert fname == "Set5_000"
    assert lr is not None and sr is not None and hr is not None
    assert "RGB" in psnr
    assert isinstance(ssim, float)


def test_benchmark_validation_epoch_end_logs_means():
    """on_validation_epoch_end consumes the buffer and emits per-dataset mean
    PSNR + SSIM via pl_module.log (verified by mocking the log method)."""
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=99)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = MagicMock()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    # Hand-populate buffer with two entries; image tensors set None so the
    # image-strip branch is skipped (covered by the next test).
    cb._buffer["Set5"] = [
        ("img_0", None, None, None, {"RGB": 30.0}, 0.9),
        ("img_1", None, None, None, {"RGB": 32.0}, 0.85),
    ]
    trainer = SimpleNamespace(global_step=42, loggers=[])
    cb.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)
    # Two log calls per dataset: psnr + ssim.
    log_keys = [call.args[0] for call in pl_module.log.call_args_list]
    assert "Set5_psnr(RGB)" in log_keys
    assert "Set5_ssim" in log_keys
    # Mean PSNR = (30.0 + 32.0) / 2 = 31.0
    psnr_call = next(c for c in pl_module.log.call_args_list if c.args[0] == "Set5_psnr(RGB)")
    assert psnr_call.args[1] == pytest.approx(31.0)


def test_benchmark_validation_epoch_end_logs_image_strips_when_on_interval(tmp_path: Path):
    """When _val_run_count hits the log interval, image strips are emitted to
    a TensorBoardLogger via experiment.add_image / add_scalar."""
    import lightning.pytorch.loggers as pl_loggers
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = MagicMock()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    cb._buffer["Set5"] = [
        ("img_0", torch.rand(3, 4, 4), torch.rand(3, 4, 4), torch.rand(3, 4, 4),
         {"RGB": 30.0}, 0.9),
    ]
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    trainer = SimpleNamespace(global_step=42, loggers=[tb_logger])
    cb.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)
    tb_logger.finalize("success")  # flush


def test_benchmark_image_strip_first_panel_is_bicubic_at_hr_size(tmp_path: Path, monkeypatch):
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

    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = MagicMock()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    lr = torch.rand(3, 4, 4)
    sr = torch.rand(3, 16, 16)
    hr = torch.rand(3, 16, 16)
    cb._buffer["Set5"] = [("img_0", lr, sr, hr, {"RGB": 30.0}, 0.9)]
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    trainer = SimpleNamespace(global_step=42, loggers=[tb_logger])
    cb.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)
    tb_logger.finalize("success")

    assert len(captured) == 1
    panels = captured[0]
    assert len(panels) == 3
    expected_bicubic = BenchmarkImageLogger._bicubic_to(lr, hr.shape[-2:])
    torch.testing.assert_close(panels[0], expected_bicubic)
    # SR is already HR-sized so _pad_to_match is a no-op; HR is untouched.
    torch.testing.assert_close(panels[1], sr)
    torch.testing.assert_close(panels[2], hr)


def test_benchmark_image_strips_handle_upscaling_sizes(tmp_path: Path):
    """For an upscaling model (SRResNet) the LR is smaller than SR/HR; the
    strip must bicubic-upsample LR to HR size rather than crash make_grid on
    a size mismatch."""
    import lightning.pytorch.loggers as pl_loggers
    cb = BenchmarkImageLogger(dataset_names=["Set5"], log_every_n_val_runs=1)
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="fit")
    pl_module = MagicMock()
    cb.on_validation_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    # LR 4x4, SR and HR 16x16 (x4 model).
    cb._buffer["Set5"] = [
        ("img_0", torch.rand(3, 4, 4), torch.rand(3, 16, 16), torch.rand(3, 16, 16),
         {"RGB": 30.0}, 0.9),
    ]
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    trainer = SimpleNamespace(global_step=42, loggers=[tb_logger])
    cb.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)  # must not raise
    tb_logger.finalize("success")


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
        trainer=trainer, pl_module=pl_module, outputs=None,
        batch=batch, batch_idx=0, dataloader_idx=1,
    )
    assert len(cb._buffer["Set14"]) == 1
    assert len(cb._buffer["Set5"]) == 0


def test_benchmark_test_epoch_end_logs_means():
    cb = BenchmarkImageLogger(dataset_names=["Set5"])
    cb.setup(SimpleNamespace(datamodule=None), pl_module=None, stage="test")
    pl_module = MagicMock()
    cb.on_test_epoch_start(trainer=SimpleNamespace(), pl_module=pl_module)
    cb._buffer["Set5"] = [
        ("img_0", None, None, None, {"RGB": 28.0}, 0.7),
    ]
    trainer = SimpleNamespace(global_step=0, loggers=[])
    cb.on_test_epoch_end(trainer=trainer, pl_module=pl_module)
    log_keys = [call.args[0] for call in pl_module.log.call_args_list]
    assert "Set5_psnr(RGB)" in log_keys


# ---------------------------------------------------------------------------
# WeightHistogramLogger on-cadence path
# ---------------------------------------------------------------------------

def test_weight_histogram_logger_emits_on_cadence(tmp_path: Path):
    import lightning.pytorch.loggers as pl_loggers
    cb = WeightHistogramLogger(log_every_n_steps=1)
    pl_module = MagicMock()
    pl_module.named_parameters = MagicMock(return_value=[
        ("model.feat.0.weight", torch.nn.Parameter(torch.rand(4, 3, 3, 3))),
        ("non_model_param", torch.nn.Parameter(torch.rand(4))),  # filtered out
    ])
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path), name="run", version="v")
    trainer = SimpleNamespace(global_step=1, loggers=[tb_logger])
    cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
    tb_logger.finalize("success")


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
        trainer=trainer, pl_module=pl_module, outputs=None,
        batch=batch, batch_idx=0, dataloader_idx=1,
    )
    assert len(cb._buffer["Set5"]) == 1
    _, _, _, _, psnr_dict, ssim = cb._buffer["Set5"][0]
    assert "RGB" in psnr_dict
    assert isinstance(ssim, float)


def test_benchmark_collect_batch_routes_through_processor():
    """SRCNN+YChannelProcessor: _collect_batch must extract Y from RGB LR before the
    model forward, then reconstruct SR back to RGB. Bypassing the processor would
    feed 3-channel RGB to a 1-channel Conv2d and crash with a shape mismatch."""
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
        trainer=trainer, pl_module=pl_module, outputs=None,
        batch=batch, batch_idx=0, dataloader_idx=1,
    )
    assert len(cb._buffer["Set5"]) == 1
    _, lr_cached, sr_cached, hr_cached, psnr_dict, ssim = cb._buffer["Set5"][0]
    # All cached tensors are full RGB (reconstruct stitched SR-Y with bicubic Cb/Cr).
    assert lr_cached.shape == (3, 16, 16)
    assert sr_cached.shape == (3, 16, 16)
    assert hr_cached.shape == (3, 16, 16)
    assert "RGB" in psnr_dict
    assert "YCbCr" in psnr_dict
    assert isinstance(ssim, float)


# ---------------------------------------------------------------------------
# BenchmarkImageLogger consumes the public predict_rgb seam + SRDataset (P2.1)
# ---------------------------------------------------------------------------

def test_benchmark_collect_batch_routes_through_predict_rgb():
    """_collect_batch must call the public pl_module.predict_rgb, and the
    buffered SR/HR must equal its output — proving no re-implemented, divergent
    forward path remains in the callback."""
    from unittest.mock import MagicMock

    pl_module = _make_real_pl_module()  # SRCNN + RGBProcessor, crop_border=0
    batch = (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16))
    with torch.no_grad():
        ref_sr, ref_hr = pl_module.predict_rgb(*batch)

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
        trainer=trainer, pl_module=pl_module, outputs=None,
        batch=batch, batch_idx=0, dataloader_idx=1,
    )

    spy.assert_called_once()
    call_args, _ = spy.call_args
    torch.testing.assert_close(call_args[0], batch[0])
    torch.testing.assert_close(call_args[1], batch[1])
    # Buffered SR/HR (uncropped, crop_border=0) match the public forward output.
    _, _, sr0, hr0, _, _ = cb._buffer["Set5"][0]
    torch.testing.assert_close(sr0, ref_sr[0].cpu())
    torch.testing.assert_close(hr0, ref_hr[0].cpu())


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
        trainer=trainer, pl_module=pl_module, outputs=None,
        batch=batch, batch_idx=0, dataloader_idx=1,
    )
    fnames = [entry[0] for entry in cb._buffer["Set5"]]
    assert fnames == [p.stem for p in ds.img_paths[:2]]  # e.g. ["img_00", "img_01"]
