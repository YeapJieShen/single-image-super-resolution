import functools
from types import SimpleNamespace
from unittest.mock import MagicMock

import lightning
import pytest
import torch

from sisr.models.base import SRModel
from sisr.models.srcnn import SRCNN, SRCNNTrainingConfig
from sisr.models.srresnet.model import SRResNet
from sisr.processors import (
    RGBProcessor,
    SRProcessor,
    YChannelProcessor,
    YCbCrProcessor,
)
from sisr.training import SREvalConfig, SRLightning, SRTrainingConfig


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
        training_config=SRTrainingConfig(),
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
        training_config=SRTrainingConfig(),
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
        training_config=SRTrainingConfig(),
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
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    lr = torch.rand(2, 3, 33, 33)
    hr = torch.rand(2, 3, 33, 33)
    loss, *_, sr_rgb, hr_cropped = lit._step((lr, hr))
    assert sr_rgb.shape == (2, 3, 21, 21)


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
        training_config=SRTrainingConfig(),
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
    assert not isinstance(out, (tuple, list)), "single optimizer when no scheduler"
    assert isinstance(out, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# test_step / build_psnr_tensors / flatten_hparams
# ---------------------------------------------------------------------------

def test_test_step_is_no_op(srcnn_rgb_lit: SRLightning):
    """test_step exists so Lightning iterates test_dataloaders; the body is a no-op."""
    lr = torch.rand(2, 3, 33, 33)
    hr = torch.rand(2, 3, 33, 33)
    out = srcnn_rgb_lit.test_step((lr, hr), batch_idx=0, dataloader_idx=0)
    assert out is None


def test_build_psnr_tensors_rgb_only(srcnn_rgb_lit: SRLightning):
    sr = torch.rand(1, 3, 4, 4)
    hr = torch.rand(1, 3, 4, 4)
    tensors = srcnn_rgb_lit._build_psnr_tensors(sr, hr)
    # eval_config.psnr_channels=['RGB'], separate_psnr=False -> only 'RGB' tracked.
    assert "RGB" in tensors


def test_build_psnr_tensors_with_separate_psnr_includes_per_channel():
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(psnr_channels=["RGB"], separate_psnr=True),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    sr = torch.rand(1, 3, 4, 4)
    hr = torch.rand(1, 3, 4, 4)
    tensors = lit._build_psnr_tensors(sr, hr)
    assert {"R", "G", "B", "RGB"} <= set(tensors)


def test_build_psnr_tensors_ycbcr_does_conversion():
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(psnr_channels=["YCbCr"]),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    sr = torch.rand(1, 3, 4, 4)
    hr = torch.rand(1, 3, 4, 4)
    tensors = lit._build_psnr_tensors(sr, hr)
    assert "YCbCr" in tensors
    sr_ycc, hr_ycc = tensors["YCbCr"]
    # YCbCr first channel (Y) must differ from input R-channel — verifies
    # conversion happened (vs. accidental identity).
    assert not torch.equal(sr_ycc, sr)


def test_flatten_hparams_handles_nested():
    flat = SRLightning._flatten_hparams({
        "a": 1,
        "b": {"x": 2, "y": [3, 4]},
        "c": None,             # None values dropped
        "d": SRCNN,            # class -> __name__
    })
    assert flat == {
        "a": 1,
        "b/x": 2,
        "b/y/0": 3,
        "b/y/1": 4,
        "d": "SRCNN",
    }


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
        training_config=SRTrainingConfig(),    # init_strategy='default' by default
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
    params_arg = args[0] if args else kwargs.get('params')
    metrics_arg = args[1] if len(args) > 1 else kwargs.get('metrics')

    expected_metrics = {f'val_psnr({k})': 0.0 for k in srcnn_rgb_lit._psnr_keys}
    expected_metrics['val_ssim'] = 0.0

    assert params_arg == srcnn_rgb_lit.hparams
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
    model = torch.nn.Conv2d(3, 3, 1)              # plain nn.Module, not SRModel
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
            processor=object(),                    # not an SRProcessor
            optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
        )


def test_step_calls_processor_extract_and_reconstruct():
    """_step routes through processor.extract (twice — input and HR) and processor.reconstruct."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    processor = MagicMock(spec=SRProcessor)
    # extract returns 3-channel passthrough; reconstruct returns its first arg.
    processor.extract.side_effect = lambda x: x
    processor.reconstruct.side_effect = lambda sr, lr: sr

    lit = SRLightning(
        model=model,
        processor=processor,
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    lr = torch.rand(2, 3, 33, 33)
    hr = torch.rand(2, 3, 33, 33)
    lit._step((lr, hr))

    # extract called twice: once for LR input, once for HR-for-loss.
    assert processor.extract.call_count == 2
    # reconstruct called once with (sr_model_out, lr_img).
    assert processor.reconstruct.call_count == 1


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


def test_saved_hparams_contain_processor_name():
    """The processor's class name is saved into Lightning hparams for TensorBoard distinction."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    lit = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(),
        eval_config=SREvalConfig(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    # Lightning's self.hparams is a flat dict after the _flatten_hparams step.
    assert lit.hparams.get("processor") == "RGBProcessor"
