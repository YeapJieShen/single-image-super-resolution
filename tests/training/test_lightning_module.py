import functools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import lightning
import pytest
import torch
import torch._dynamo
import torchmetrics

from sisr.losses import SRLoss
from sisr.metrics.scoring import SRScorer
from sisr.models.srcnn import SRCNN, SRCNNEvalConfig, SRCNNTrainingConfig
from sisr.models.srresnet import SRResNetEvalConfig, SRResNetTrainingConfig
from sisr.models.srresnet.model import SRResNet
from sisr.processors import (
    RGBProcessor,
    RGBSignedOutputProcessor,
    SRProcessor,
    YCbCrProcessor,
    YChannelProcessor,
)
from sisr.training import SRDataModule, SREvalConfig, SRLightning, SRTrainingConfig

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def srcnn_rgb_lit() -> SRLightning:
    """SRLightning wrapping a 3-channel SRCNN with the RGB pass-through processor."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


@pytest.fixture
def srcnn_y_lit() -> SRLightning:
    """SRLightning wrapping a 1-channel SRCNN with the Y-channel processor (paper-faithful)."""
    model = SRCNN(num_channels=1, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    return SRLightning(
        model=model,
        processor=YChannelProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(crop_border=0, psnr_channels=["RGB", "YCbCr"]),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


@pytest.fixture
def rgb_lr_hr_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A (lr, hr) batch — both 3-channel RGB, shape (2, 3, 33, 33), in [0, 1]."""
    g = torch.Generator().manual_seed(42)
    lr = torch.rand(2, 3, 33, 33, generator=g)
    hr = torch.rand(2, 3, 33, 33, generator=g)
    return lr, hr


# ---------------------------------------------------------------------------
# forward + step
# ---------------------------------------------------------------------------


def test_forward_delegates_to_model(srcnn_rgb_lit: SRLightning):
    x = torch.zeros(1, 3, 33, 33)
    out = srcnn_rgb_lit(x)
    assert out.shape == (1, 3, 21, 21)


def test_step_rgb_path(srcnn_rgb_lit: SRLightning, rgb_lr_hr_batch):
    lr, hr = rgb_lr_hr_batch
    loss, lr_in, hr_in, sr_rgb, hr_cropped = srcnn_rgb_lit._step((lr, hr))
    # SRCNN with valid padding: 33 -> 21
    assert sr_rgb.shape == (2, 3, 21, 21)
    assert hr_cropped.shape == (2, 3, 21, 21)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_step_y_path_reconstructs_rgb(srcnn_y_lit: SRLightning, rgb_lr_hr_batch):
    """Y-channel SRCNN: model takes 1-channel Y, output stitched with bicubic
    Cb/Cr from LR YCbCr to produce 3-channel SR_RGB."""
    lr, hr = rgb_lr_hr_batch
    loss, lr_in, hr_in, sr_rgb, hr_cropped = srcnn_y_lit._step((lr, hr))
    assert sr_rgb.shape == (2, 3, 21, 21), "SR_RGB is reconstructed to 3 channels"
    assert hr_cropped.shape == (2, 3, 21, 21)
    assert torch.isfinite(loss)


def test_step_y_path_loss_on_y_only():
    """With YChannelProcessor, criterion sees 1-channel inputs."""
    model = SRCNN(num_channels=1, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    criterion = MagicMock(return_value=torch.tensor(0.0, requires_grad=True))
    lit = SRLightning(
        model=model,
        processor=YChannelProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        criterion=criterion,
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    lr = torch.rand(2, 3, 33, 33)
    hr = torch.rand(2, 3, 33, 33)
    lit._step((lr, hr))
    args, _ = criterion.call_args
    sr_arg, hr_arg = args
    assert sr_arg.shape[1] == 1, "criterion should see 1-channel SR (Y)"
    assert hr_arg.shape[1] == 1, "criterion should see 1-channel HR (Y)"


def test_step_ycbcr_path():
    """3-channel YCbCr training path via YCbCrProcessor."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=YCbCrProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    lr = torch.rand(2, 3, 33, 33)
    hr = torch.rand(2, 3, 33, 33)
    loss, *_, sr_rgb, hr_cropped = lit._step((lr, hr))
    assert sr_rgb.shape == (2, 3, 21, 21)


# ---------------------------------------------------------------------------
# need_sr_rgb=False — training_step's reconstruct skip (measured: reconstruct
# costs ~11.5% of a data-free SRCNN step for YChannelProcessor, ~0 for
# SRResNet's identity/elementwise processors; training_step never consumes
# sr_rgb, so paying for it there was pure waste).
# ---------------------------------------------------------------------------


def test_step_need_sr_rgb_false_skips_reconstruct(srcnn_y_lit: SRLightning, rgb_lr_hr_batch):
    """The exact call training_step makes: reconstruct must not run, and
    sr_rgb must come back None rather than a silently-wrong placeholder."""
    lr, hr = rgb_lr_hr_batch
    spy = MagicMock(wraps=srcnn_y_lit.processor.reconstruct)
    srcnn_y_lit.processor.reconstruct = spy

    loss, _, _, sr_rgb, hr_cropped = srcnn_y_lit._step((lr, hr), need_sr_rgb=False)

    spy.assert_not_called()
    assert sr_rgb is None
    assert hr_cropped.shape == (2, 3, 21, 21)
    assert torch.isfinite(loss)


def test_step_default_still_reconstructs(srcnn_y_lit: SRLightning, rgb_lr_hr_batch):
    """need_sr_rgb defaults to True — validation/predict/direct callers must
    keep getting the exact reconstruction this project has always produced."""
    lr, hr = rgb_lr_hr_batch
    spy = MagicMock(wraps=srcnn_y_lit.processor.reconstruct)
    srcnn_y_lit.processor.reconstruct = spy

    loss, _, _, sr_rgb, hr_cropped = srcnn_y_lit._step((lr, hr))

    spy.assert_called_once()
    assert sr_rgb is not None
    assert sr_rgb.shape == (2, 3, 21, 21)


def test_step_loss_bit_identical_with_and_without_reconstruct_skip(
    srcnn_y_lit: SRLightning, rgb_lr_hr_batch
):
    """The loss is computed from sr_model_out/hr_cropped alone — never sr_rgb
    — so skipping reconstruct must not move it by even a rounding error."""
    lr, hr = rgb_lr_hr_batch

    loss_with, *_ = srcnn_y_lit._step((lr, hr))
    loss_without, *_ = srcnn_y_lit._step((lr, hr), need_sr_rgb=False)

    assert torch.equal(loss_with, loss_without)


def test_forward_sr_need_sr_rgb_false_hr_crop_matches_sr_model_out_size(
    srcnn_rgb_lit: SRLightning, rgb_lr_hr_batch
):
    """hr_cropped must still land at the correct spatial size (derived from
    sr_model_out, not sr_rgb, when reconstruct is skipped)."""
    lr, hr = rgb_lr_hr_batch
    sr_model_out, sr_rgb, hr_cropped = srcnn_rgb_lit._forward_sr(lr, hr, need_sr_rgb=False)
    assert sr_rgb is None
    assert hr_cropped.shape[-2:] == sr_model_out.shape[-2:] == (21, 21)


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


def test_configure_optimizers_uniform(srcnn_rgb_lit: SRLightning):
    out = srcnn_rgb_lit.configure_optimizers()
    opt = out[0][0] if isinstance(out, tuple) else out
    assert len(opt.param_groups) == 1, "uniform LR should produce one param_group"
    assert opt.param_groups[0]["lr"] == 1e-4


def test_configure_optimizers_per_layer_lrs():
    """Per-Conv2d param_groups with explicit absolute LRs."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(layer_lrs=[1.0e-4, 1.0e-4, 1.0e-5]),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4, momentum=0.9),
    )
    out = lit.configure_optimizers()
    opt = out[0][0] if isinstance(out, tuple) else out
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs == [1e-4, 1e-4, 1e-5]


def test_configure_optimizers_per_blob_lrs():
    """A 2-element entry splits a Conv2d into separate weight and bias LRs.

    The SRCNN authors' prototxt carries two ``param`` blocks per layer
    (weights, then bias), so conv1/conv2 biases train at a tenth of their
    weights' rate. One LR per Conv2d cannot express that.
    """
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(
            layer_lrs=[[1.0e-4, 1.0e-5], [1.0e-4, 1.0e-5], [1.0e-5, 1.0e-5]]
        ),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4, momentum=0.9),
    )
    out = lit.configure_optimizers()
    opt = out[0][0] if isinstance(out, tuple) else out

    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    lr_of = {}
    for group in opt.param_groups:
        for param in group["params"]:
            lr_of[id(param)] = group["lr"]

    assert [lr_of[id(c.weight)] for c in convs] == [1e-4, 1e-4, 1e-5]
    assert [lr_of[id(c.bias)] for c in convs] == [1e-5, 1e-5, 1e-5]


def test_configure_optimizers_scalar_and_pair_entries_mix():
    """A bare float still means 'weight and bias together', alongside pairs."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(layer_lrs=[1.0e-4, [1.0e-4, 1.0e-5], 1.0e-5]),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    out = lit.configure_optimizers()
    opt = out[0][0] if isinstance(out, tuple) else out

    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}

    assert [lr_of[id(c.weight)] for c in convs] == [1e-4, 1e-4, 1e-5]
    assert [lr_of[id(c.bias)] for c in convs] == [1e-4, 1e-5, 1e-5]


def test_configure_optimizers_per_blob_wrong_pair_length_raises():
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(layer_lrs=[[1e-4, 1e-5, 1e-6], 1e-4, 1e-5]),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    with pytest.raises(ValueError, match="exactly 2"):
        lit.configure_optimizers()


def test_configure_optimizers_per_blob_bias_lr_without_bias_raises():
    """A bias LR aimed at a bias-free Conv2d is a config error, not a silent drop."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    convs[1].bias = None
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(layer_lrs=[1e-4, [1e-4, 1e-5], 1e-5]),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    with pytest.raises(ValueError, match="has no bias"):
        lit.configure_optimizers()


def test_configure_optimizers_per_layer_count_mismatch_raises():
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(layer_lrs=[1e-4, 1e-4]),  # 2 vs 3
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    with pytest.raises(ValueError, match="Conv2d count"):
        lit.configure_optimizers()


def test_configure_optimizers_per_layer_with_non_conv_params_raises():
    """SRResNet has BatchNorm/PReLU; layer_lrs=[...] must reject because
    not every trainable param lives inside a Conv2d."""
    model = SRResNet(scale=2, num_residual_blocks=1)
    n_convs = sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(layer_lrs=[1e-4] * n_convs),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    with pytest.raises(ValueError, match="trainable param"):
        lit.configure_optimizers()


def test_configure_optimizers_with_lr_scheduler(srcnn_rgb_lit: SRLightning):
    """When lr_scheduler is provided, returns ([opt], [sched])."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        lr_scheduler=functools.partial(torch.optim.lr_scheduler.StepLR, step_size=10),
    )
    out = lit.configure_optimizers()
    assert isinstance(out, tuple) or isinstance(out, list)
    opts, scheds = out
    assert len(opts) == 1
    assert len(scheds) == 1
    assert isinstance(scheds[0], torch.optim.lr_scheduler.StepLR)


def test_configure_optimizers_no_scheduler_returns_bare(srcnn_rgb_lit: SRLightning):
    out = srcnn_rgb_lit.configure_optimizers()
    assert not isinstance(out, tuple | list), "single optimizer when no scheduler"
    assert isinstance(out, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# test_step / build_metric_tensors / flatten_hparams
# ---------------------------------------------------------------------------


def test_test_step_is_no_op(srcnn_rgb_lit: SRLightning):
    """test_step exists so Lightning iterates test_dataloaders; the body is a no-op."""
    lr = torch.rand(2, 3, 33, 33)
    hr = torch.rand(2, 3, 33, 33)
    out = srcnn_rgb_lit.test_step((lr, hr), batch_idx=0, dataloader_idx=0)
    assert out is None


def test_build_metric_tensors_rgb_only_when_neither_metric_requests_y():
    """Only the RGB family is built when both psnr_channels and ssim_channels
    are pinned to ['RGB'] — unlike the base SREvalConfig default, which also
    pulls in 'Y' via ssim_channels (see the union test below)."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(psnr_channels=["RGB"], ssim_channels=["RGB"]),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    sr = torch.rand(1, 3, 4, 4)
    hr = torch.rand(1, 3, 4, 4)
    tensors = lit.scorer.metric_tensors(sr, hr)
    assert set(tensors) == {"RGB", "R", "G", "B"}


def test_build_metric_tensors_union_of_psnr_and_ssim_keys():
    """Regression: a colorspace requested only by ssim_channels (not
    psnr_channels) must still get a tensor entry — the tensor map is built
    from the *union* of psnr_keys and ssim_keys, not psnr_keys alone, so
    SSIM-only keys aren't silently missing."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(psnr_channels=["RGB"], ssim_channels=["RGB", "Y"]),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    sr = torch.rand(1, 3, 4, 4)
    hr = torch.rand(1, 3, 4, 4)
    tensors = lit.scorer.metric_tensors(sr, hr)
    assert "Y" in tensors


def test_build_metric_tensors_with_separate_psnr_includes_per_channel():
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(psnr_channels=["RGB"], separate_psnr=True),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    sr = torch.rand(1, 3, 4, 4)
    hr = torch.rand(1, 3, 4, 4)
    tensors = lit.scorer.metric_tensors(sr, hr)
    assert {"R", "G", "B", "RGB"} <= set(tensors)


def test_build_metric_tensors_ycbcr_does_conversion():
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(psnr_channels=["YCbCr"]),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    sr = torch.rand(1, 3, 4, 4)
    hr = torch.rand(1, 3, 4, 4)
    tensors = lit.scorer.metric_tensors(sr, hr)
    assert "YCbCr" in tensors
    sr_ycc, hr_ycc = tensors["YCbCr"]
    # YCbCr first channel (Y) must differ from input R-channel — verifies
    # conversion happened (vs. accidental identity).
    assert not torch.equal(sr_ycc, sr)


def test_build_metric_tensors_ycbcr_uses_studio_range_not_full_range():
    """Regression: the metric-side YCbCr conversion must be BT.601
    studio range, not the full-range conversion SRProcessor subclasses train
    in — locked by comparing against sisr.colorspace directly."""
    from sisr.colorspace import rgb_to_ycbcr, rgb_to_ycbcr_studio

    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(psnr_channels=["YCbCr"]),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    sr = torch.rand(1, 3, 4, 4, generator=torch.Generator().manual_seed(3))
    hr = torch.rand(1, 3, 4, 4, generator=torch.Generator().manual_seed(4))
    tensors = lit.scorer.metric_tensors(sr, hr)
    sr_ycc, hr_ycc = tensors["YCbCr"]
    torch.testing.assert_close(sr_ycc, rgb_to_ycbcr_studio(sr))
    torch.testing.assert_close(hr_ycc, rgb_to_ycbcr_studio(hr))
    assert not torch.allclose(sr_ycc, rgb_to_ycbcr(sr))


def test_flatten_hparams_handles_nested():
    flat = SRLightning._flatten_hparams(
        {
            "a": 1,
            "b": {"x": 2, "y": [3, 4]},
            "c": None,  # None values dropped
            "d": SRCNN,  # class -> __name__
        }
    )
    assert flat == {
        "a": 1,
        "b/x": 2,
        "b/y/0": 3,
        "b/y/1": 4,
        "d": "SRCNN",
    }


def test_flatten_hparams_stringifies_a_value_of_no_other_kind():
    """The catch-all is the only thing standing between an unusual config
    field and a crash while writing the TensorBoard HParams tab.

    No shipped config reaches it today -- every field is a primitive, a list,
    a dict or a class -- which is why it sat uncovered. It is reachable by any
    architecture whose config holds something else (a Path, an enum), so it is
    a live safety net rather than dead code, and it is cheaper to pin than to
    remove and rediscover.
    """
    flat = SRLightning._flatten_hparams({"where": Path("/tmp/x"), "nested": {"k": Path("y")}})

    assert flat == {"where": str(Path("/tmp/x")), "nested/k": str(Path("y"))}


# ---------------------------------------------------------------------------
# init strategy
# ---------------------------------------------------------------------------


def test_paper_init_calls_reset_parameters():
    """init_strategy='paper' → SRLightning calls model.reset_parameters(...)."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    model.reset_parameters = MagicMock()
    SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRCNNTrainingConfig(init_strategy="paper", init_mean=0.5, init_std=0.02),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    model.reset_parameters.assert_called_once_with(mean=0.5, std=0.02)


def test_default_init_skips_reset_parameters():
    """init_strategy='default' → SRLightning does NOT call model.reset_parameters(...)."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    model.reset_parameters = MagicMock()
    SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRCNNTrainingConfig(init_strategy="default"),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    model.reset_parameters.assert_not_called()


def test_base_training_config_skips_reset_parameters():
    """Base SRTrainingConfig defaults init_strategy='default' → no call, no crash."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    model.reset_parameters = MagicMock()
    SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),  # init_strategy='default' by default
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    model.reset_parameters.assert_not_called()


# ---------------------------------------------------------------------------
# on_train_start hook
# ---------------------------------------------------------------------------


def test_on_train_start_logs_hparams_with_val_metrics(srcnn_rgb_lit: SRLightning):
    """The hook calls TensorBoardLogger.log_hyperparams once with val metrics dict."""
    tb = MagicMock(spec=lightning.pytorch.loggers.TensorBoardLogger)
    srcnn_rgb_lit.trainer = SimpleNamespace(loggers=[tb])

    srcnn_rgb_lit.on_train_start()

    tb.log_hyperparams.assert_called_once()
    args, kwargs = tb.log_hyperparams.call_args
    params_arg = args[0] if args else kwargs.get("params")
    metrics_arg = args[1] if len(args) > 1 else kwargs.get("metrics")

    expected_metrics = {f"psnr/val/{k}": 0.0 for k in srcnn_rgb_lit.eval_config.psnr_keys}
    expected_metrics.update({f"ssim/val/{k}": 0.0 for k in srcnn_rgb_lit.eval_config.ssim_keys})

    # The TB-only flattened view (_tb_hparams), not self.hparams — the latter must stay
    # nested/unflattened so checkpoints round-trip through --ckpt_path (see lightning_module.py).
    assert params_arg == srcnn_rgb_lit._tb_hparams
    assert metrics_arg == expected_metrics


def test_on_train_start_no_tb_logger_is_noop(srcnn_rgb_lit: SRLightning):
    """The hook does nothing when no TensorBoardLogger is attached."""
    csv = MagicMock(spec=lightning.pytorch.loggers.CSVLogger)
    srcnn_rgb_lit.trainer = SimpleNamespace(loggers=[csv])

    srcnn_rgb_lit.on_train_start()

    csv.log_hyperparams.assert_not_called()


def test_on_train_start_no_loggers_is_noop(srcnn_rgb_lit: SRLightning):
    """The hook does nothing when the loggers list is empty."""
    srcnn_rgb_lit.trainer = SimpleNamespace(loggers=[])

    srcnn_rgb_lit.on_train_start()  # must not raise


def test_on_train_start_multiple_tb_loggers_each_receive_call(srcnn_rgb_lit: SRLightning):
    """When multiple TensorBoardLoggers are attached, each gets log_hyperparams."""
    tb1 = MagicMock(spec=lightning.pytorch.loggers.TensorBoardLogger)
    tb2 = MagicMock(spec=lightning.pytorch.loggers.TensorBoardLogger)
    srcnn_rgb_lit.trainer = SimpleNamespace(loggers=[tb1, tb2])

    srcnn_rgb_lit.on_train_start()

    tb1.log_hyperparams.assert_called_once()
    tb2.log_hyperparams.assert_called_once()


# ---------------------------------------------------------------------------
# new: isinstance guards and processor flow
# ---------------------------------------------------------------------------


def test_srlightning_rejects_non_srmodel():
    """SRLightning(model=<plain nn.Module>) raises TypeError with a readable message."""
    model = torch.nn.Conv2d(3, 3, 1)  # plain nn.Module, not SRModel
    with pytest.raises(TypeError, match="SRModel subclass"):
        SRLightning(
            model=model,
            processor=RGBProcessor(),
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )


def test_srlightning_rejects_non_srprocessor():
    """SRLightning(processor=<not SRProcessor>) raises TypeError with a readable message."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    with pytest.raises(TypeError, match="SRProcessor subclass"):
        SRLightning(
            model=model,
            processor=object(),  # not an SRProcessor
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )


def test_srlightning_construction_rejects_mismatched_num_channels():
    """Regression: SRCNNTrainingConfig.validate_against catches a
    num_channels/processor mismatch at construction, not silently at train time."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    with pytest.raises(ValueError, match="num_channels"):
        SRLightning(
            model=model,
            processor=YChannelProcessor(),  # model_channels=1, mismatches num_channels=3
            training_config=SRCNNTrainingConfig(),
            eval_config=SREvalConfig(),
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )


def test_srlightning_construction_rejects_mismatched_in_out_channels():
    """Same regression for SRResNet's in_out_channels/processor correlation."""
    model = SRResNet(scale=2, num_residual_blocks=1, in_out_channels=3)
    with pytest.raises(ValueError, match="in_out_channels"):
        SRLightning(
            model=model,
            processor=YChannelProcessor(),  # model_channels=1, mismatches in_out_channels=3
            training_config=SRResNetTrainingConfig(),
            eval_config=SREvalConfig(),
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )


def test_srlightning_construction_accepts_matching_channels():
    """The happy path (matching num_channels/processor) must not raise."""
    model = SRCNN(num_channels=1, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    SRLightning(
        model=model,
        processor=YChannelProcessor(),
        training_config=SRCNNTrainingConfig(),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )  # must not raise


def test_srlightning_construction_rejects_mismatched_example_input_shape_channels():
    """The base (architecture-agnostic) channel check fires even with a plain
    SRTrainingConfig, via example_input_shape[0] vs processor.model_channels."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    with pytest.raises(ValueError, match="example_input_shape"):
        SRLightning(
            model=model,
            processor=RGBProcessor(),  # model_channels=3
            training_config=SRTrainingConfig(example_input_shape=(1, 33, 33)),  # 1 != 3
            eval_config=SREvalConfig(),
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )


def test_srlightning_construction_forward_probe_succeeds_with_matching_shape():
    """A correctly-paired example_input_shape runs the real forward probe with
    no error — the base validate_against no-ops past the channel check."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(example_input_shape=(3, 33, 33)),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )  # must not raise


def test_step_calls_processor_extract_and_reconstruct():
    """_step routes LR through extract, HR through extract_target, output through reconstruct."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    processor = MagicMock(spec=SRProcessor)
    # extract returns 3-channel passthrough; reconstruct returns its first arg.
    processor.extract.side_effect = lambda x: x
    processor.extract_target.side_effect = lambda x: x
    processor.reconstruct.side_effect = lambda sr, lr: sr

    lit = SRLightning(
        model=model,
        processor=processor,
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    lr = torch.rand(2, 3, 33, 33)
    hr = torch.rand(2, 3, 33, 33)
    lit._step((lr, hr))

    # extract handles only the LR input; the HR loss target goes through
    # extract_target, so an asymmetric processor can transform them differently.
    assert processor.extract.call_count == 1
    assert processor.extract_target.call_count == 1
    # reconstruct called once with (sr_model_out, lr_img).
    assert processor.reconstruct.call_count == 1


def test_step_loss_uses_extract_target_not_extract():
    """The loss target is extract_target(hr) — proven by a processor where they differ."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBSignedOutputProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    lr = torch.rand(2, 3, 33, 33)
    hr = torch.rand(2, 3, 33, 33)

    loss, _, _, sr_rgb, hr_cropped = lit._step((lr, hr))

    # Reproduce the loss by hand against the [-1, 1] target. Scoring `sr_rgb`
    # against `hr_cropped` (the [0, 1] pair the metrics use) would give exactly
    # a quarter of this — which is what a regression to extract() would log.
    sr_model_out = 2.0 * sr_rgb - 1.0
    expected = torch.nn.functional.mse_loss(sr_model_out, hr_cropped * 2.0 - 1.0)
    assert torch.allclose(loss, expected, atol=1e-6)
    assert not torch.allclose(loss, expected / 4.0, atol=1e-6)


def test_paper_init_polymorphic_on_non_overriding_subclass():
    """init_strategy='paper' on a SRModel subclass without paper init is a no-op (not a crash)."""
    # SRResNet doesn't override reset_parameters; the base SRModel.reset_parameters
    # accepts **kwargs and does nothing.
    model = SRResNet(scale=2, num_residual_blocks=1)
    SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(init_strategy="paper", init_mean=0.5, init_std=0.02),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    # No exception = success. (The actual weights are whatever PyTorch's default init produced.)


def test_saved_tb_hparams_contain_processor_name():
    """The processor's class name is saved into the TB-only hparams view for TensorBoard."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    assert lit._tb_hparams.get("processor") == "RGBProcessor"
    # Regression guard: self.hparams (what checkpoints save) must NOT carry this —
    # it's ignored on purpose so --ckpt_path reload never has to reparse it.
    assert "processor" not in lit.hparams


# ---------------------------------------------------------------------------
# predict_rgb public inference seam
# ---------------------------------------------------------------------------


def test_predict_rgb_returns_sr_and_hr_pair(srcnn_rgb_lit: SRLightning, rgb_lr_hr_batch):
    lr, hr = rgb_lr_hr_batch
    sr_rgb, hr_cropped = srcnn_rgb_lit.predict_rgb(lr, hr)
    # SRCNN with valid padding: 33 -> 21; HR center-cropped to match.
    assert sr_rgb.shape == (2, 3, 21, 21)
    assert hr_cropped.shape == (2, 3, 21, 21)


def test_predict_rgb_always_reconstructs_regardless_of_need_sr_rgb_default(
    srcnn_rgb_lit: SRLightning, rgb_lr_hr_batch
):
    """predict_rgb (the BenchmarkImageLogger/scoring seam) never passes
    need_sr_rgb=False — its whole contract is returning a real sr_rgb."""
    lr, hr = rgb_lr_hr_batch
    spy = MagicMock(wraps=srcnn_rgb_lit.processor.reconstruct)
    srcnn_rgb_lit.processor.reconstruct = spy
    sr_rgb, _ = srcnn_rgb_lit.predict_rgb(lr, hr)
    spy.assert_called_once()
    assert sr_rgb is not None


def test_predict_rgb_matches_step_forward_path(srcnn_rgb_lit: SRLightning, rgb_lr_hr_batch):
    """predict_rgb must produce the exact sr_rgb / hr_cropped that _step
    returns — both route through the single _forward_sr core, so the training
    and benchmark-logging paths cannot diverge."""
    lr, hr = rgb_lr_hr_batch
    _, _, _, step_sr, step_hr = srcnn_rgb_lit._step((lr, hr))
    pred_sr, pred_hr = srcnn_rgb_lit.predict_rgb(lr, hr)
    torch.testing.assert_close(pred_sr, step_sr)
    torch.testing.assert_close(pred_hr, step_hr)


def test_predict_rgb_matches_step_forward_path_y_channel(srcnn_y_lit: SRLightning, rgb_lr_hr_batch):
    """Same invariant on the Y-channel path (reconstruct stitches SR-Y with
    bicubic Cb/Cr) — the reconstructed RGB must be identical across paths."""
    lr, hr = rgb_lr_hr_batch
    _, _, _, step_sr, step_hr = srcnn_y_lit._step((lr, hr))
    pred_sr, pred_hr = srcnn_y_lit.predict_rgb(lr, hr)
    torch.testing.assert_close(pred_sr, step_sr)
    torch.testing.assert_close(pred_hr, step_hr)


# ---------------------------------------------------------------------------
# Metric-path clamp to [0, 1], applied once in _forward_lr
# ---------------------------------------------------------------------------


def _overshooting_rgb_lit() -> SRLightning:
    """SRCNN + RGBProcessor (identity reconstruct) with the tail conv biased
    hugely positive, so sr_rgb genuinely overshoots [0, 1] on any input — the
    scenario the metric-path clamp exists for."""
    model = SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same")
    torch.nn.init.constant_(model.recon.bias, 5.0)
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def test_step_clamps_sr_rgb_but_not_the_loss_target():
    """Regression: sr_rgb returned by _step is clamped to [0, 1] even
    though the model genuinely overshoots, while the loss — read from
    sr_model_out, never sr_rgb — is exactly the unclamped MSE. A clamp that
    leaked into the loss would kill gradients on saturated pixels (constraint
    1) and would also make this hand-computed loss equal the "wrong_loss"
    below instead."""
    lit = _overshooting_rgb_lit()
    lr = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    hr = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(1))

    sr_model_out, _ = lit._forward_lr(lr)
    assert sr_model_out.max() > 1.0, "fixture must genuinely overshoot [0, 1]"

    loss, _, _, sr_rgb, hr_cropped = lit._step((lr, hr))

    assert sr_rgb.min() >= 0.0
    assert sr_rgb.max() <= 1.0

    expected_loss = torch.nn.functional.mse_loss(sr_model_out, hr_cropped)
    torch.testing.assert_close(loss, expected_loss)

    wrong_loss = torch.nn.functional.mse_loss(sr_rgb, hr_cropped)
    assert not torch.allclose(loss, wrong_loss)


def test_step_psnr_matches_hand_clamped_reference_not_raw_output():
    """Regression: PSNR scored on the validation_step path must equal
    a hand-clamped reference computed from the raw model output, not the PSNR
    the unclamped output would give — proving overshoot is being penalised
    away rather than silently scored."""
    lit = _overshooting_rgb_lit()
    lr = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    hr = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(1))

    sr_model_out, _ = lit._forward_lr(lr)
    _, _, _, sr_rgb, hr_cropped = lit._step((lr, hr))

    scored_psnr = SRScorer.psnr(sr_rgb, hr_cropped)
    hand_clamped_psnr = SRScorer.psnr(sr_model_out.clamp(0.0, 1.0), hr_cropped)
    unclamped_psnr = SRScorer.psnr(sr_model_out, hr_cropped)

    torch.testing.assert_close(scored_psnr, hand_clamped_psnr)
    assert not torch.allclose(scored_psnr, unclamped_psnr)


def test_predict_rgb_clamps_for_benchmark_logger_consumer():
    """Regression: predict_rgb — the seam BenchmarkImageLogger reads
    for benchmark/test-set metrics — must return the same clamped sr_rgb as
    the validation_step path, not a re-implemented, unclamped forward."""
    lit = _overshooting_rgb_lit()
    lr = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    hr = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(1))

    sr_rgb, _ = lit.predict_rgb(lr, hr)
    assert sr_rgb.min() >= 0.0
    assert sr_rgb.max() <= 1.0


def test_predict_step_output_also_clamped_via_shared_forward_lr():
    """predict_step shares _forward_lr with _forward_sr (see its docstring),
    so the same clamp covers it as a side effect — locks that the seam really
    is shared rather than duplicated across the two call sites."""
    lit = _overshooting_rgb_lit()
    lr = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    out = lit.predict_step(lr, batch_idx=0)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# No dead stateful metric accumulator
# ---------------------------------------------------------------------------


def test_val_metrics_hold_no_stateful_accumulators(srcnn_y_lit: SRLightning):
    """Regression: PSNR/SSIM are computed via torchmetrics.functional,
    so SRLightning registers no stateful torchmetrics.Metric accumulators that
    would grow unread and unreset across a validation run."""
    stateful = [m for m in srcnn_y_lit.modules() if isinstance(m, torchmetrics.Metric)]
    assert stateful == [], (
        f"expected no stateful torchmetrics.Metric accumulators; found {stateful}"
    )


# ---------------------------------------------------------------------------
# Nested config dataclasses expand into HParams columns
# ---------------------------------------------------------------------------


def test_tb_hparams_expand_nested_config_fields():
    """Regression: training_config/eval_config dataclasses expand into
    individual HParams columns (via dataclasses.asdict) using the '/' separator,
    instead of a single stringified blob under the bare key — in the TB-only
    flattened view, since self.hparams itself must stay nested (see below)."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(crop_border=7),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    flat = dict(lit._tb_hparams)
    assert flat.get("eval_config/crop_border") == 7
    assert flat.get("eval_config/ssim_impl") == "wang"
    assert flat.get("training_config/init_strategy") == "default"
    # regression guard: no stringified dataclass blob under the bare key
    assert "eval_config" not in flat
    assert "training_config" not in flat


def test_hparams_stay_nested_plain_dicts_for_checkpoint_reload():
    """Regression: self.hparams (what Lightning writes into the checkpoint's
    `hyper_parameters`) must stay a plain nested dict — no '/'-flattening and
    no live SRTrainingConfig/SREvalConfig objects. LightningCLI._parse_ckpt_path
    re-parses this dict as CLI options (model.<key>), so a '/' in a key breaks
    every --ckpt_path invocation, and a live dataclass object there breaks
    weights_only=True checkpoint loading."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(crop_border=7),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    assert lit.hparams["eval_config"] == {
        "crop_border": 7,
        "psnr_channels": ["RGB"],
        "separate_psnr": False,
        "ssim_channels": ["RGB", "Y"],
        "ssim_impl": "wang",
        "perceptual_metrics": [],
        "lpips_net": "alex",
    }
    assert isinstance(lit.hparams["training_config"], dict)
    assert not any("/" in k for k in lit.hparams)


# ---------------------------------------------------------------------------
# val PSNR is the per-image mean, invariant to batch size
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# predict_step — LR-only inference seam
# ---------------------------------------------------------------------------


def test_forward_lr_matches_forward_sr_sr_component(srcnn_rgb_lit: SRLightning, rgb_lr_hr_batch):
    """_forward_sr's (sr_model_out, sr_rgb) must equal _forward_lr's output on
    the same LR — the refactor shares the pipeline instead of forking it."""
    lr, hr = rgb_lr_hr_batch
    sr_model_out_a, sr_rgb_a, _ = srcnn_rgb_lit._forward_sr(lr, hr)
    sr_model_out_b, sr_rgb_b = srcnn_rgb_lit._forward_lr(lr)
    torch.testing.assert_close(sr_model_out_a, sr_model_out_b)
    torch.testing.assert_close(sr_rgb_a, sr_rgb_b)


def test_predict_step_matches_forward_lr_rgb_path(srcnn_rgb_lit: SRLightning):
    lr = torch.rand(2, 3, 33, 33, generator=torch.Generator().manual_seed(0))
    _, expected_sr = srcnn_rgb_lit._forward_lr(lr)
    out = srcnn_rgb_lit.predict_step(lr, batch_idx=0)
    torch.testing.assert_close(out, expected_sr)


def test_predict_step_y_channel_reconstructs_rgb(srcnn_y_lit: SRLightning):
    """Y-channel model output is 1-channel; predict_step must still return
    RGB — the processor.reconstruct step stitches back bicubic LR Cb/Cr.
    srcnn_y_lit uses 'valid' padding (33 -> 21), so the size check locks that
    same shrinkage; the channel check is the actual regression guard."""
    lr = torch.rand(2, 3, 33, 33, generator=torch.Generator().manual_seed(1))
    out = srcnn_y_lit.predict_step(lr, batch_idx=0)
    assert out.shape == (2, 3, 21, 21)


def test_predict_step_srresnet_upsamples_by_scale():
    """Genuine-LR RGB path: output must be exactly scale x the LR input."""
    model = SRResNet(scale=2, num_residual_blocks=1)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    lr = torch.rand(1, 3, 12, 12)
    out = lit.predict_step(lr, batch_idx=0)
    assert out.shape == (1, 3, 24, 24)


def test_predict_step_dataloader_idx_default_is_ignored(srcnn_rgb_lit: SRLightning):
    """A single predict loader is expected; dataloader_idx must not change output."""
    lr = torch.rand(1, 3, 33, 33, generator=torch.Generator().manual_seed(2))
    out_default = srcnn_rgb_lit.predict_step(lr, batch_idx=0)
    out_explicit = srcnn_rgb_lit.predict_step(lr, batch_idx=0, dataloader_idx=0)
    torch.testing.assert_close(out_default, out_explicit)


def test_val_psnr_is_per_image_mean_not_batch_pooled():
    """Regression: PSNR for a batch equals the mean of the per-image
    PSNRs (SR-standard reduction, invariant to val batch_size), not the
    batch-pooled PSNR that a default-dim PeakSignalNoiseRatio would give."""
    from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_fn

    g = torch.Generator().manual_seed(1)
    hr = torch.rand(2, 3, 8, 8, generator=g) * 0.8  # keep values in [0, 0.8]
    sr = hr.clone()
    sr[0] = sr[0] + 0.01  # image 0: small error
    sr[1] = sr[1] + 0.15  # image 1: larger error

    per_image = torch.stack(
        [psnr_fn(sr[i : i + 1], hr[i : i + 1], data_range=1.0) for i in range(2)]
    ).mean()
    pooled = psnr_fn(sr, hr, data_range=1.0)  # dim=None -> pools the batch

    batch_val = SRScorer.psnr(sr, hr)
    assert torch.allclose(batch_val, per_image, atol=1e-5)
    assert not torch.allclose(batch_val, pooled, atol=1e-3)


def _make_lit_with_ssim_impl(ssim_impl: str) -> SRLightning:
    """Small RGB SRCNN wrapped in SRLightning, varying only ``eval_config.ssim_impl``."""
    model = SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same")
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(crop_border=0, ssim_channels=["Y"], ssim_impl=ssim_impl),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def test_mean_ssim_dispatches_on_eval_config():
    """One seam decides which SSIM exists. Flipping ssim_impl must change the
    value, or the flag is silently inert."""
    import sisr.metrics.ssim

    sr = torch.rand(1, 1, 64, 64, generator=torch.Generator().manual_seed(0))
    hr = torch.rand(1, 1, 64, 64, generator=torch.Generator().manual_seed(1))

    wang = _make_lit_with_ssim_impl("wang")
    daala = _make_lit_with_ssim_impl("daala")

    assert daala.scorer.ssim(sr, hr).item() == pytest.approx(
        sisr.metrics.ssim.daala_ssim(sr, hr).item(), rel=1e-12
    )
    assert wang.scorer.ssim(sr, hr).item() != pytest.approx(daala.scorer.ssim(sr, hr).item())


def test_mean_ssim_uses_daala_through_real_srresnet_eval_config():
    """Coverage gap: every other ssim_impl test above builds a manually
    constructed base SREvalConfig, never SRResNet's own (unconfigured)
    SRResNetEvalConfig() — so a future change that special-cases by eval-config
    subclass, or a CLI/subclass field-resolution bug, could flip the
    architecture default without failing anything. This computes a real SSIM
    through the real subclass, mirroring test_metadata.py's _make_srresnet_lit()
    construction rather than a bespoke one.

    torch.manual_seed seeds SRResNet's weight init for reproducibility, even
    though SRScorer.ssim never runs the model forward — sr/hr are independent
    inputs — so nothing here is actually seed-sensitive, but a prior task's
    unseeded model was flagged in review and this follows the same discipline.
    """
    from torchmetrics.functional.image import structural_similarity_index_measure

    import sisr.metrics.ssim

    torch.manual_seed(0)
    model = SRResNet(scale=4, hidden_channel=8, num_residual_blocks=1)
    lit = SRLightning(
        model=model,
        processor=RGBSignedOutputProcessor(),
        training_config=SRResNetTrainingConfig(),
        eval_config=SRResNetEvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )

    sr = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(2))
    hr = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(3))

    result = lit.scorer.ssim(sr, hr)

    assert result.item() == pytest.approx(sisr.metrics.ssim.daala_ssim(sr, hr).item(), rel=1e-12)
    wang_result = structural_similarity_index_measure(sr, hr, data_range=1.0)
    assert result.item() != pytest.approx(wang_result.item())


# ---------------------------------------------------------------------------
# compile_backend — configurable torch.compile plumbing
# ---------------------------------------------------------------------------


def _make_compilable_lit(compile_backend: str | None, example_input_shape=(3, 8, 8)):
    """Small SRCNN wrapped in SRLightning; 'eager'/'aot_eager' run on CPU and
    are bit-identical to uncompiled, so this is CI-testable without a GPU."""
    model = SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same")
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(
            compile_backend=compile_backend, example_input_shape=example_input_shape
        ),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def test_compile_backend_invalid_name_raises_at_construction():
    """A backend name torch._dynamo doesn't recognize fails immediately at
    SRLightning construction — a YAML typo fails at startup, not mid-run."""
    model = SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same")
    with pytest.raises(torch._dynamo.exc.InvalidBackend):
        SRLightning(
            model=model,
            processor=RGBProcessor(),
            training_config=SRTrainingConfig(compile_backend="not_a_real_backend"),
            eval_config=SREvalConfig(),
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )


def test_compile_backend_none_leaves_compiled_unset():
    lit = _make_compilable_lit(compile_backend=None)
    assert lit._compiled is None


def test_compile_backend_state_dict_has_no_compiled_or_orig_mod_keys():
    """Regression: torch.compile() wraps the model in an OptimizedModule, an
    nn.Module — a plain attribute assignment would register it as a
    submodule and duplicate every parameter under '_compiled._orig_mod.*'
    in state_dict(), the checkpoint-compatibility hazard this wiring exists
    to avoid."""
    lit = _make_compilable_lit(compile_backend="eager")
    keys = list(lit.state_dict().keys())
    assert keys == [
        "model.feat.0.weight",
        "model.feat.0.bias",
        "model.mapping.0.weight",
        "model.mapping.0.bias",
        "model.recon.weight",
        "model.recon.bias",
    ]
    assert not any("_compiled" in k or "_orig_mod" in k for k in keys)


def test_compile_backend_shares_weights_no_copy():
    """self.model and self._compiled._orig_mod are the exact same object —
    one set of weights, two execution paths, no drift."""
    lit = _make_compilable_lit(compile_backend="eager")
    assert lit._compiled._orig_mod is lit.model


def test_compile_backend_checkpoint_roundtrips_compiled_to_plain():
    """A checkpoint saved with compile_backend set loads strict=True into a
    module built with it unset — the wiring must be invisible to state_dict."""
    compiled_lit = _make_compilable_lit(compile_backend="eager")
    plain_lit = _make_compilable_lit(compile_backend=None)
    plain_lit.load_state_dict(compiled_lit.state_dict(), strict=True)  # must not raise


def test_compile_backend_checkpoint_roundtrips_plain_to_compiled():
    """The reverse direction: a checkpoint saved with compilation off loads
    strict=True into a module built with compile_backend set."""
    plain_lit = _make_compilable_lit(compile_backend=None)
    compiled_lit = _make_compilable_lit(compile_backend="eager")
    compiled_lit.load_state_dict(plain_lit.state_dict(), strict=True)  # must not raise


def test_compile_backend_eager_output_matches_uncompiled_exactly():
    """Regression: the 'eager' backend is bit-identical to uncompiled, which
    is what makes this wiring testable on CPU without a GPU."""
    torch.manual_seed(0)
    plain_lit = _make_compilable_lit(compile_backend=None)
    torch.manual_seed(0)
    compiled_lit = _make_compilable_lit(compile_backend="eager")
    compiled_lit.load_state_dict(plain_lit.state_dict(), strict=True)

    x = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out_plain = plain_lit.model(x)
        out_compiled = compiled_lit._compiled(x)
    torch.testing.assert_close(out_plain, out_compiled, rtol=0, atol=0)


def test_compile_backend_train_mode_dispatches_to_compiled_callable():
    """In train mode, _forward_lr must call through self._compiled, not
    self.model directly — proven by replacing each with an independent
    tracking stub so neither call site can accidentally reach the other."""
    lit = _make_compilable_lit(compile_backend="eager")
    lit.train()

    compiled_mock = MagicMock(return_value=torch.zeros(1, 3, 8, 8))
    object.__setattr__(lit, "_compiled", compiled_mock)
    lit.model.forward = MagicMock(return_value=torch.zeros(1, 3, 8, 8))

    lit._forward_lr(torch.rand(1, 3, 8, 8))

    compiled_mock.assert_called_once()
    lit.model.forward.assert_not_called()


def test_compile_backend_eval_mode_dispatches_to_eager_model():
    """In eval mode, _forward_lr must call self.model directly and never
    touch self._compiled — required because validation/predict see widely
    varying image sizes that a static-shape backend can't handle."""
    lit = _make_compilable_lit(compile_backend="eager")
    lit.eval()

    compiled_mock = MagicMock(return_value=torch.zeros(1, 3, 8, 8))
    object.__setattr__(lit, "_compiled", compiled_mock)
    lit.model.forward = MagicMock(return_value=torch.zeros(1, 3, 8, 8))

    lit._forward_lr(torch.rand(1, 3, 8, 8))

    lit.model.forward.assert_called_once()
    compiled_mock.assert_not_called()


def test_compile_backend_off_uses_eager_model_regardless_of_training_flag():
    """compile_backend=None must route through self.model in both modes —
    there is no compiled callable to ever dispatch to."""
    lit = _make_compilable_lit(compile_backend=None)
    x = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))

    lit.train()
    out_train, _ = lit._forward_lr(x)
    lit.eval()
    out_eval, _ = lit._forward_lr(x)
    torch.testing.assert_close(out_train, out_eval)


# ---------------------------------------------------------------------------
# compile_mode -- the inductor mode a YAML run could not previously select
# ---------------------------------------------------------------------------


def test_compile_mode_reaches_torch_compile(monkeypatch):
    """The whole point of the field: a mode set in YAML must arrive at
    `torch.compile`.

    Every published `reduce-overhead` / `max-autotune` figure described something
    no config could select, because the call site passed `backend` and nothing
    else -- so a configured run silently got inductor's default mode.
    """
    seen = {}

    def spy(model, **kwargs):
        # Records rather than compiles: a real inductor compile here would only
        # add a slow, toolchain-dependent step to a question about argument
        # passing. That inductor accepts the mode is pinned separately, by
        # test_compile_mode_unrecognised_by_torch_raises_at_construction.
        seen.update(kwargs)
        return model

    model = SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same")
    monkeypatch.setattr(torch, "compile", spy)
    SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(
            compile_backend="inductor", compile_mode="reduce-overhead"
        ),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    assert seen["mode"] == "reduce-overhead"


def test_compile_mode_defaults_to_none_so_compiled_runs_are_unchanged(monkeypatch):
    """Adding the field must not renumber any existing compiled run: unset means
    `mode=None`, which is what `torch.compile` already defaulted to."""
    seen = {}
    real_compile = torch.compile

    def spy(model, **kwargs):
        seen.update(kwargs)
        return real_compile(model, **kwargs)

    monkeypatch.setattr(torch, "compile", spy)
    _make_compilable_lit(compile_backend="eager")
    assert seen["mode"] is None


@pytest.mark.parametrize("backend", [None, "cudagraphs", "aot_eager"])
def test_compile_mode_without_the_inductor_backend_is_refused(backend):
    """`mode` is an inductor concept. With no backend nothing is compiled and the
    mode is dead config; with another backend torch forwards `mode` as a keyword
    to a compiler function that does not take one, which fails on the first
    compiled call -- a mid-run crash, which is exactly what this project's
    compile plumbing exists to convert into a startup one.
    """
    with pytest.raises(ValueError, match="compile_mode"):
        SRTrainingConfig(compile_backend=backend, compile_mode="reduce-overhead")


def test_compile_mode_unrecognised_by_torch_raises_at_construction():
    """A typo'd mode must fail at `SRLightning` construction, like a typo'd
    backend already does -- not 12 hours into a run."""
    model = SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same")
    with pytest.raises(RuntimeError, match="mode"):
        SRLightning(
            model=model,
            processor=RGBProcessor(),
            training_config=SRTrainingConfig(
                compile_backend="inductor", compile_mode="not_a_real_mode"
            ),
            eval_config=SREvalConfig(),
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )


def test_on_fit_start_warms_up_compiled_path():
    """on_fit_start must actually invoke the compiled callable once, so a
    missing-toolchain backend fails at fit-start rather than mid-run."""
    lit = _make_compilable_lit(compile_backend="eager", example_input_shape=(3, 8, 8))
    calls = []
    object.__setattr__(lit, "_compiled", lambda x: calls.append(x) or lit.model(x))

    lit.on_fit_start()

    assert len(calls) == 1
    assert calls[0].shape == (1, 3, 8, 8)


def test_on_fit_start_noop_when_compile_backend_unset():
    lit = _make_compilable_lit(compile_backend=None)
    lit.on_fit_start()  # must not raise


def test_on_fit_start_noop_when_example_input_shape_unset():
    """No shape to probe with — must not raise, not fabricate an input."""
    lit = _make_compilable_lit(compile_backend="eager", example_input_shape=None)
    lit.on_fit_start()  # must not raise


# ---------------------------------------------------------------------------
# on_save_checkpoint — sisr_meta provenance (public Lightning hook)
# ---------------------------------------------------------------------------


def test_on_save_checkpoint_injects_sisr_meta(srcnn_rgb_lit: SRLightning):
    checkpoint = {"global_step": 500, "epoch": 2, "state_dict": srcnn_rgb_lit.state_dict()}
    srcnn_rgb_lit.on_save_checkpoint(checkpoint)
    meta = checkpoint["sisr_meta"]
    assert meta["format"] == "sisr-meta-v2"
    assert meta["model"]["class_path"] == "sisr.models.srcnn.model.SRCNN"
    assert meta["training"]["global_step"] == 500
    assert meta["training"]["epoch"] == 2


def test_on_save_checkpoint_leaves_monitor_unset():
    """A full .ckpt isn't tied to any one monitored metric (zero, one, or
    several SRCheckpoint callbacks may independently watch it), unlike
    SRWeightsCheckpoint's per-save monitor/monitor_value."""
    lit = SRLightning(
        model=SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    checkpoint = {"global_step": 1, "epoch": 0}
    lit.on_save_checkpoint(checkpoint)
    assert checkpoint["sisr_meta"]["training"]["monitor"] is None
    assert checkpoint["sisr_meta"]["training"]["monitor_value"] is None


def test_on_save_checkpoint_does_not_touch_other_checkpoint_keys(srcnn_rgb_lit: SRLightning):
    """Only adds sisr_meta — must not mutate hyper_parameters (the key
    LightningCLI._parse_ckpt_path actually reads for --ckpt_path resumption)
    or any other existing entry."""
    checkpoint = {
        "global_step": 1,
        "epoch": 0,
        "hyper_parameters": {"foo": "bar"},
        "pytorch-lightning_version": "2.6.5",
    }
    srcnn_rgb_lit.on_save_checkpoint(checkpoint)
    assert checkpoint["hyper_parameters"] == {"foo": "bar"}
    assert checkpoint["pytorch-lightning_version"] == "2.6.5"
    assert "sisr_meta" in checkpoint


def test_on_save_checkpoint_round_trips_weights_only(tmp_path, srcnn_rgb_lit: SRLightning):
    """The whole checkpoint, sisr_meta included, must survive
    torch.load(..., weights_only=True) — the safe-loading contract every
    consumer (including LightningCLI's own ckpt_path parsing) depends on."""
    checkpoint = {
        "global_step": 10,
        "epoch": 0,
        "state_dict": srcnn_rgb_lit.state_dict(),
    }
    srcnn_rgb_lit.on_save_checkpoint(checkpoint)

    path = tmp_path / "test.ckpt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, weights_only=True)

    assert loaded["sisr_meta"] == checkpoint["sisr_meta"]
    assert set(loaded["state_dict"].keys()) == set(srcnn_rgb_lit.state_dict().keys())


# ---------------------------------------------------------------------------
# setup() — input_contract probe + example_input_shape probe (real datasets)
# ---------------------------------------------------------------------------


def _srcnn_datamodule(tiny_rgb_image_dir: Path, tmp_path: Path, scale: int = 2) -> SRDataModule:
    """SRDataModule over srcnn's pre-upsampled datasets (lr.shape == hr.shape)."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": scale,
            "use_tqdm": False,
            "cache_dir": str(tmp_path / ".lmdb_cache_srcnn"),
            "build_num_workers": 1,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": scale},
    }
    return SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


def _srresnet_datamodule(
    tiny_rgb_image_dir: Path, tmp_path: Path, scale: int = 2, hr_crop_size: int = 24
) -> SRDataModule:
    """SRDataModule over srresnet's native-LR datasets (hr.shape == lr.shape * scale)."""
    train_spec = {
        "class_path": "sisr.datasets.srresnet.TrainDataset",
        "init_args": {
            "img_dir": str(tiny_rgb_image_dir),
            "scale": scale,
            "hr_crop_size": hr_crop_size,
            "use_tqdm": False,
            "cache_dir": str(tmp_path / ".lmdb_cache_srresnet"),
            "build_num_workers": 1,
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srresnet.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": scale},
    }
    return SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


def _srresnet_lit(scale: int = 2) -> SRLightning:
    model = SRResNet(scale=scale, num_residual_blocks=1)
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRResNetTrainingConfig(scale=scale),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def _srcnn_lit() -> SRLightning:
    model = SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def test_setup_raises_on_native_lr_model_with_pre_upsampled_dataset(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """Regression: SRResNet (native_lr) wired to SRCNN's pre-upsampled dataset
    must raise at setup(), not silently corrupt training via _forward_sr's
    center_crop zero-padding."""
    lit = _srresnet_lit(scale=2)
    dm = _srcnn_datamodule(tiny_rgb_image_dir, tmp_path, scale=2)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    with pytest.raises(ValueError, match="input_contract"):
        lit.setup(stage="fit")


def test_setup_raises_on_pre_upsampled_model_with_native_lr_dataset(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """Mirror of the above: SRCNN (pre_upsampled) wired to SRResNet's
    native-LR dataset must also raise."""
    lit = _srcnn_lit()
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    with pytest.raises(ValueError, match="input_contract"):
        lit.setup(stage="fit")


def test_setup_accepts_correctly_paired_native_lr_model_and_dataset(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    lit = _srresnet_lit(scale=2)
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="fit")  # must not raise


def test_setup_accepts_correctly_paired_pre_upsampled_model_and_dataset(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    lit = _srcnn_lit()
    dm = _srcnn_datamodule(tiny_rgb_image_dir, tmp_path, scale=2)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="fit")  # must not raise


def test_setup_checks_validate_stage_too_not_just_fit(tiny_rgb_image_dir: Path, tmp_path: Path):
    """Regression: the probe must cover validate/test, not only fit — a
    validate-only run (no train dataset instantiated at all) must still
    catch a cross-wired model/dataset pairing via the primary val dataset."""
    lit = _srresnet_lit(scale=2)
    dm = _srcnn_datamodule(tiny_rgb_image_dir, tmp_path, scale=2)
    dm.setup(stage="validate")
    assert dm.train_dataset is None, "validate stage must not build the train dataset"
    lit.trainer = SimpleNamespace(datamodule=dm)

    with pytest.raises(ValueError, match="input_contract"):
        lit.setup(stage="validate")


def test_setup_skips_pair_check_for_predict_only_datamodule(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """PredictDataset has no HR pair; a predict-only datamodule (train/val/test
    all unbuilt) must not raise even for a would-be-mismatched model."""
    lit = _srresnet_lit(scale=2)
    train_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
    }
    dm = SRDataModule(
        train_dataset=train_spec,
        val_dataset=train_spec,
        predict_dataset={
            "class_path": "sisr.datasets.predict.PredictDataset",
            "init_args": {"img_dir": str(tiny_rgb_image_dir)},
        },
    )
    dm.setup(stage="predict")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="predict")  # must not raise


def test_setup_noop_when_trainer_unattached(srcnn_rgb_lit: SRLightning):
    """Guard: pure-module unit-test construction (no Trainer attached) must
    not raise — self._trainer is None."""
    srcnn_rgb_lit.setup(stage="fit")  # must not raise


def test_setup_noop_when_datamodule_unset():
    """Guard: a Trainer attached but with no datamodule must not raise."""
    lit = _srcnn_lit()
    lit.trainer = SimpleNamespace(datamodule=None)
    lit.setup(stage="fit")  # must not raise


def test_setup_raises_on_wrong_example_input_shape(tiny_rgb_image_dir: Path, tmp_path: Path):
    """Regression: training_config.example_input_shape spatial dims must
    match the real train LR patch — currently unvalidated (validate_against
    only checks the channel count)."""
    lit = _srresnet_lit(scale=2)
    lit.training_config.example_input_shape = (3, 999, 999)  # wrong H/W
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    with pytest.raises(ValueError, match="example_input_shape"):
        lit.setup(stage="fit")


def test_setup_accepts_correct_example_input_shape(tiny_rgb_image_dir: Path, tmp_path: Path):
    """hr_crop_size=24 // scale=2 == 12: the real train LR patch side."""
    lit = _srresnet_lit(scale=2)
    lit.training_config.example_input_shape = (3, 12, 12)
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="fit")  # must not raise


def test_setup_skips_example_input_shape_check_when_unset(tiny_rgb_image_dir: Path, tmp_path: Path):
    lit = _srresnet_lit(scale=2)
    assert lit.training_config.example_input_shape is None
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="fit")  # must not raise


def test_setup_skips_example_input_shape_check_when_no_train_dataset(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """example_input_shape is train-only — a validate run (no train dataset
    instantiated) must not check it, even when it's set and would mismatch
    the (variable-size) val images."""
    lit = _srresnet_lit(scale=2)
    lit.training_config.example_input_shape = (3, 999, 999)
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="validate")
    assert dm.train_dataset is None
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="validate")  # must not raise


# ---------------------------------------------------------------------------
# _forward_sr belt-and-braces: HR must not be smaller than SR (center_crop
# would otherwise zero-pad silently instead of cropping)
# ---------------------------------------------------------------------------


def test_forward_sr_raises_when_hr_smaller_than_sr():
    """A native-LR model whose SR output outgrows the HR it's paired with
    must raise before torchvision.center_crop would zero-pad it."""
    lit = _srresnet_lit(scale=4)
    lr = torch.rand(1, 3, 8, 8)  # sr_rgb will be 32x32
    hr = torch.rand(1, 3, 16, 16)  # smaller than sr_rgb in both dims

    with pytest.raises(ValueError, match="center_crop|zero-pad|smaller"):
        lit._forward_sr(lr, hr)


def test_forward_sr_accepts_hr_larger_than_sr_srcnn_direction(srcnn_rgb_lit: SRLightning):
    """SRCNN's legitimate direction — HR strictly larger than SR (valid-padding
    shrinkage) — must keep working unchanged."""
    lr = torch.rand(1, 3, 33, 33)
    hr = torch.rand(1, 3, 33, 33)
    _, sr_rgb, hr_cropped = srcnn_rgb_lit._forward_sr(lr, hr)
    assert sr_rgb.shape == (1, 3, 21, 21)
    assert hr_cropped.shape == (1, 3, 21, 21)


def test_forward_sr_accepts_hr_exactly_equal_to_sr():
    """HR exactly equal to SR (the common same-padding / native-LR-correct
    case) must not raise — only strictly smaller HR is a problem."""
    lit = _srresnet_lit(scale=2)
    lr = torch.rand(1, 3, 8, 8)
    hr = torch.rand(1, 3, 16, 16)  # exactly lr * scale
    _, sr_rgb, hr_cropped = lit._forward_sr(lr, hr)
    assert sr_rgb.shape == hr_cropped.shape == (1, 3, 16, 16)


# ---------------------------------------------------------------------------
# Code-review follow-ups: model.hparams['scale'] fallback, RNG isolation,
# foreign-datamodule degradation, unrecognised input_contract
# ---------------------------------------------------------------------------


def test_setup_raises_using_model_scale_fallback_when_training_config_scale_unset(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """Regression: a native_lr model paired with a bare SRTrainingConfig()
    (scale=None — the construction several existing tests use) must still
    catch a scale-mismatched dataset via the model's own 'scale' hparam,
    instead of silently skipping the check just because
    training_config.scale wasn't set. Model declares scale=2; the dataset
    actually downsamples by 4."""
    model = SRResNet(scale=2, num_residual_blocks=1)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),  # scale=None
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=4, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    with pytest.raises(ValueError, match="input_contract"):
        lit.setup(stage="fit")


def test_setup_still_skips_when_both_training_config_and_model_scale_absent(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """When neither training_config.scale nor model.hparams['scale'] is
    set, there is genuinely nothing to check against — must not raise."""
    model = SRResNet(scale=2, num_residual_blocks=1)
    del model.hparams["scale"]  # simulate a native_lr model with no scale hparam at all
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),  # scale=None
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=4, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="fit")  # must not raise


def test_setup_samples_train_dataset_index_zero_only_once(
    tiny_rgb_image_dir: Path, tmp_path: Path, monkeypatch
):
    """When train_dataset is both the input_contract probe target and the
    example_input_shape check's source, __getitem__(0) must be read once
    and reused — not sampled twice.

    Wraps with a plain function (not a MagicMock instance) assigned onto the
    class: CPython's implicit special-method dispatch for `dataset[0]` only
    passes `self` through when the found `__getitem__` is a real function
    object — a Mock instance stored there is invoked without `self`, which
    would silently miscount (or crash) here.
    """
    lit = _srresnet_lit(scale=2)
    lit.training_config.example_input_shape = (3, 12, 12)
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    ds_cls = type(dm.train_dataset)
    original_getitem = ds_cls.__getitem__
    calls: list[int] = []

    def counting_getitem(self, idx):
        calls.append(idx)
        return original_getitem(self, idx)

    monkeypatch.setattr(ds_cls, "__getitem__", counting_getitem)

    lit.setup(stage="fit")

    assert calls == [0]


def test_setup_does_not_perturb_global_random_state(tiny_rgb_image_dir: Path, tmp_path: Path):
    """Regression: TrainDataset.__getitem__'s random.randint crop draws must
    not leak into the global random sequence the real training loop consumes
    — this is a paper-reproduction repo, so an unguarded probe call would
    silently shift every seeded crop drawn after setup()."""
    import random

    lit = _srresnet_lit(scale=2)
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    random.seed(12345)
    state_before = random.getstate()
    lit.setup(stage="fit")
    assert random.getstate() == state_before


def test_setup_skips_when_datamodule_lacks_dataset_accessors():
    """A foreign datamodule (not an SRDataModule — exposes none of
    train_dataset/val_dataset/test_datasets) must degrade to 'nothing to
    probe' instead of raising AttributeError."""
    lit = _srcnn_lit()
    lit.trainer = SimpleNamespace(datamodule=SimpleNamespace())

    lit.setup(stage="fit")  # must not raise


def test_check_input_contract_raises_on_unrecognised_contract(srcnn_rgb_lit: SRLightning):
    """A future/typo'd input_contract value must raise clearly, instead of
    silently falling into the native_lr branch (the old bare if/else)."""
    srcnn_rgb_lit.model.input_contract = "not_a_real_contract"
    lr = torch.rand(1, 3, 8, 8)
    hr = torch.rand(1, 3, 8, 8)

    with pytest.raises(ValueError, match="not_a_real_contract"):
        srcnn_rgb_lit._check_input_contract(lr, hr, "train_dataset", object())


# ---------------------------------------------------------------------------
# Real-world regression: the probe must not poison the live LMDB-backed train
# dataset for pickling (num_workers > 0 DataLoaders spawn, unconditionally on
# Windows, and must pickle the dataset to send it to worker processes).
# ---------------------------------------------------------------------------


def test_setup_leaves_train_dataset_picklable_for_spawned_workers(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """Regression: setup()'s probe must not call __getitem__(0) on the live
    dm.train_dataset. Doing so lazily opens LMDBCache._env (an
    lmdb.Environment, not picklable) on that exact instance — the same
    instance a num_workers > 0 DataLoader later hands to spawned worker
    processes via pickle. Before the fix, pickle.dumps(dm.train_dataset)
    raised TypeError: cannot pickle 'Environment' object after setup() ran.
    """
    import pickle

    lit = _srresnet_lit(scale=2)
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="fit")

    pickle.dumps(dm.train_dataset)  # must not raise


def test_setup_gracefully_reprobes_dataset_already_opened_by_real_training(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """Regression: a second setup() call (e.g. `test` after `fit` in the same
    process — trainer.fit() then trainer.test() on the same datamodule, a
    real scenario test_integration.py already exercises) against a
    train_dataset whose LMDB env real training already opened for real
    (harmless on its own for num_workers=0 -- nothing pickles a num_workers=0
    dataset) must not raise. The picklability guarantee _sample_zero
    provides is "this probe never opens a still-pristine dataset first", not
    "the dataset stays picklable forever no matter what real training does
    to it afterwards" -- a naive implementation that hard-raises whenever
    the pickle round trip fails would reject this legitimate re-probe."""
    lit = _srresnet_lit(scale=2)
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="fit")
    dm.train_dataset[0]  # simulate a real in-process (num_workers=0) training read

    lit.setup(stage="fit")  # must not raise on the re-probe


# ---------------------------------------------------------------------------
# _extra_probe — subclass hook fed setup()'s already-sampled (lr, hr) pair
# ---------------------------------------------------------------------------


def test_setup_calls_extra_probe_with_the_sampled_pair(tiny_rgb_image_dir: Path, tmp_path: Path):
    """A subclass hook must receive the exact (lr, hr) pair setup() already
    sampled from the real train dataset — not just any call. Asserting only
    "was called" can't tell a correct pair from an empty or wrong one, so
    this checks the actual shapes and their scale relationship."""
    seen = {}

    class Probing(SRLightning):
        def _extra_probe(self, lr, hr, source):
            seen["lr_shape"] = tuple(lr.shape)
            seen["hr_shape"] = tuple(hr.shape)
            seen["source"] = source

    model = SRResNet(scale=2, num_residual_blocks=1)
    lit = Probing(
        model=model,
        processor=RGBProcessor(),
        training_config=SRResNetTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="fit")

    assert seen["source"] == "train_dataset"
    assert seen["hr_shape"][-1] == seen["lr_shape"][-1] * 2


# ---------------------------------------------------------------------------
# criterion binding
# ---------------------------------------------------------------------------


class _SpyLoss(SRLoss):
    """Records every bind() call so wiring can be asserted, not assumed."""

    def __init__(self):
        super().__init__()
        self.bound: list[SRProcessor] = []

    def bind(self, processor: SRProcessor) -> None:
        self.bound.append(processor)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).pow(2).mean()


def _lit_with_criterion(criterion, processor):
    return SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=processor,
        eval_config=SREvalConfig(crop_border=0),
        criterion=criterion,
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def test_srloss_criterion_is_bound_once_to_the_modules_own_processor():
    """The bind seam is the only way a loss learns output_range/model_channels.
    Binding twice would let a stateful loss double-apply its range mapping."""
    spy, processor = _SpyLoss(), RGBSignedOutputProcessor()

    _lit_with_criterion(spy, processor)

    assert spy.bound == [processor], "bind must run exactly once, with this module's processor"


def test_plain_nn_module_criterion_is_never_bound_and_still_trains():
    """Regression guard for the isinstance branch: torch.nn losses have no bind()
    and must keep working untouched, including the MSELoss default."""
    lit = _lit_with_criterion(torch.nn.L1Loss(), RGBProcessor())
    lr, hr = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)

    loss, *_ = lit._step((lr, hr))

    assert torch.isfinite(loss)
    assert lit.criterion_description == "L1Loss"


def test_criterion_description_prefers_describe_for_srloss():
    """One derivation point feeds both the HParams column and checkpoint metadata."""

    class _Described(_SpyLoss):
        def describe(self) -> str:
            return "spy(recipe)"

    lit = _lit_with_criterion(_Described(), RGBProcessor())

    assert lit.criterion_description == "spy(recipe)"
    assert lit._tb_hparams["criterion"] == "spy(recipe)"


# ---------------------------------------------------------------------------
# validation_step — perceptual metric logging (lpips/dists)
# ---------------------------------------------------------------------------


def build_module(eval_config: SREvalConfig | None = None) -> SRLightning:
    """Native-LR SRLightning (SRResNet-style, scale=4) for validation_step-level tests.

    lr (1, 3, 8, 8) -> sr (1, 3, 32, 32), matching the hr size these tests use, so
    validation_step's HR-not-smaller-than-SR check never fires.
    """
    model = SRResNet(scale=4, num_residual_blocks=1)
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=4),
        eval_config=eval_config or SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def capture_logged_metrics(module: SRLightning) -> dict[str, float]:
    """Monkeypatch module.log to record name -> float(value), bypassing the real
    Trainer-attached self.log (which validation_step's callers would otherwise need).

    Detaches tensors before the float() cast — validation_step runs outside
    torch.no_grad() here (no real Trainer to enter it), so logged tensors carry
    requires_grad, and float() on those warns (an error under filterwarnings).
    """
    logged: dict[str, float] = {}

    def fake_log(name, value, **kwargs):
        logged[name] = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)

    module.log = fake_log
    return logged


def test_validation_logs_perceptual_tags(monkeypatch):
    """Requested perceptual metrics reach TensorBoard under their own tag family."""
    monkeypatch.setattr(
        "sisr.metrics.scoring.perceptual_score",
        lambda name, sr, hr, lpips_net: torch.tensor(0.5 if name == "lpips" else 0.25),
    )
    module = build_module(eval_config=SREvalConfig(perceptual_metrics=["lpips", "dists"]))
    logged = capture_logged_metrics(module)

    module.validation_step((torch.rand(1, 3, 8, 8), torch.rand(1, 3, 32, 32)), 0)

    assert logged["lpips/val"] == pytest.approx(0.5)
    assert logged["dists/val"] == pytest.approx(0.25)


def test_no_perceptual_tags_when_unrequested(monkeypatch):
    """Default-off must mean no new tags for existing SRCNN/SRResNet runs."""
    module = build_module(eval_config=SREvalConfig())
    logged = capture_logged_metrics(module)
    module.validation_step((torch.rand(1, 3, 8, 8), torch.rand(1, 3, 32, 32)), 0)
    assert not [tag for tag in logged if tag.startswith(("lpips", "dists"))]


def test_example_input_shape_is_derived_from_the_real_train_sample(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """A value the code already checks is a value the code can supply.

    It used to be written out by hand under two different rules -- crop size over
    scale for a native-LR dataset, sub-image size for a pre-upsampled one -- and
    then validated against the very tensor those rules describe.
    """
    lit = SRLightning(
        model=SRResNet(scale=2, num_residual_blocks=1),
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    assert lit.training_config.example_input_shape is None
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    lit.setup(stage="fit")

    # hr_crop_size 24 over scale 2, in 3 channels -- never stated anywhere.
    assert lit.training_config.example_input_shape == (3, 12, 12)
    assert tuple(lit.example_input_array.shape) == (1, 3, 12, 12)


def test_an_explicit_example_input_shape_is_still_checked_not_overwritten(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """Stating an intent and having it verified is the reason the field survives."""
    lit = SRLightning(
        model=SRResNet(scale=2, num_residual_blocks=1),
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2, example_input_shape=(3, 99, 99)),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    dm = _srresnet_datamodule(tiny_rgb_image_dir, tmp_path, scale=2, hr_crop_size=24)
    dm.setup(stage="fit")
    lit.trainer = SimpleNamespace(datamodule=dm)

    with pytest.raises(ValueError, match="example_input_shape"):
        lit.setup(stage="fit")


def test_subclass_crop_border_follows_the_models_scale_not_a_constant():
    """A paper-default eval config crops `scale` pixels at ANY scale, not a fixed 3 or 4.

    The SR field's convention -- and the SRCNN authors' own demo code -- crop the
    outer ``scale`` pixels before scoring. `SRResNetEvalConfig` shipped a hardcoded
    ``4`` and `SRCNNEvalConfig` a hardcoded ``3``, which are correct only because
    those templates happen to ship x4 and x3. At any other scale the constant is
    silently the wrong border, and a border changes the reported PSNR/SSIM without
    changing the model.
    """
    model = SRResNet(scale=2, num_residual_blocks=1, hidden_channel=4)
    lit = SRLightning(
        model=model,
        processor=RGBSignedOutputProcessor(),
        training_config=SRResNetTrainingConfig(scale=2),
        eval_config=SRResNetEvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    assert lit.eval_config.crop_border == 2, (
        "a scale-2 run must crop 2 border pixels, not the shipped constant"
    )
    assert lit.scorer.eval_config.crop_border == 2, "the scorer must see the resolved border"


def test_explicit_crop_border_still_wins_over_the_scale_default():
    """An explicit value is an intent, and is never overwritten by the derivation."""
    model = SRResNet(scale=2, num_residual_blocks=1, hidden_channel=4)
    lit = SRLightning(
        model=model,
        processor=RGBSignedOutputProcessor(),
        training_config=SRResNetTrainingConfig(scale=2),
        eval_config=SRResNetEvalConfig(crop_border=8),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    assert lit.eval_config.crop_border == 8


def test_derived_crop_border_refuses_when_no_scale_exists_to_derive_from():
    """The sentinel needs a scale. With none available, say so rather than guess.

    SRCNN declares no ``scale`` hparam of its own, so a bare
    ``SRTrainingConfig()`` leaves nothing to derive from. Silently falling back
    to 0 would score the full image while the config asked for the field
    convention -- a different measurement with nothing to indicate it.
    """
    model = SRCNN(num_channels=1, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    with pytest.raises(ValueError, match="no scale is available"):
        SRLightning(
            model=model,
            processor=YChannelProcessor(),
            training_config=SRTrainingConfig(),
            eval_config=SREvalConfig(crop_border=None),
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )


def test_scorer_refuses_an_unresolved_crop_border_sentinel():
    """A scorer built outside SRLightning must not silently treat None as 0."""
    scorer = SRScorer(SREvalConfig(crop_border=None))
    with pytest.raises(ValueError, match="still None at scoring time"):
        scorer.crop(torch.rand(1, 3, 16, 16))


def test_eval_padding_makes_the_scored_region_the_authors():
    """With the authors' inference path the SR field's border is all that is lost.

    Valid convolution costs 6 px per side before `crop_border` shaves any,
    so the scored region sat strictly inside the authors'. Under
    `eval_padding='same'` the model output is full-size, `_forward_sr`'s
    center_crop becomes a no-op, and `crop_border` alone decides the region
    — which #217 already derives from `scale`.
    """
    model = SRCNN(
        num_channels=3,
        num_filters=(64, 32),
        kernel_sizes=(9, 1, 5),
        padding=0,
        eval_padding="same",
    )
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRCNNTrainingConfig(scale=3),
        eval_config=SRCNNEvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    lit.eval()
    assert lit.eval_config.crop_border == 3  # from scale, not a pinned constant

    hr = torch.rand(1, 3, 48, 48)
    with torch.no_grad():
        sr, hr_aligned = lit.predict_rgb(hr.clone(), hr)

    assert sr.shape == hr.shape, "same-padded inference must not shrink the SR field"
    assert hr_aligned.shape == hr.shape, "so center_crop must have nothing left to take"
