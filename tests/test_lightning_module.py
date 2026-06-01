"""Unit tests for SRLightning hooks."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import lightning
import torch

from sisr.training import SRLightning


def _make_module() -> SRLightning:
    """Minimal SRLightning instance for hook unit tests."""
    model = torch.nn.Conv2d(3, 3, 3, padding=1)
    return SRLightning(model=model)


def test_on_train_start_logs_hparams_with_val_metrics():
    """The hook calls TensorBoardLogger.log_hyperparams once with val metrics dict."""
    module = _make_module()
    tb = MagicMock(spec=lightning.pytorch.loggers.TensorBoardLogger)
    module.trainer = SimpleNamespace(loggers=[tb])

    module.on_train_start()

    tb.log_hyperparams.assert_called_once()
    args, kwargs = tb.log_hyperparams.call_args
    params_arg = args[0] if args else kwargs.get('params')
    metrics_arg = args[1] if len(args) > 1 else kwargs.get('metrics')

    expected_metrics = {f'val_psnr({k})': 0.0 for k in module._psnr_keys}
    expected_metrics['val_ssim'] = 0.0

    assert params_arg == module.hparams
    assert metrics_arg == expected_metrics


def test_on_train_start_no_tb_logger_is_noop():
    """The hook does nothing when no TensorBoardLogger is attached."""
    module = _make_module()
    csv = MagicMock(spec=lightning.pytorch.loggers.CSVLogger)
    module.trainer = SimpleNamespace(loggers=[csv])

    module.on_train_start()

    csv.log_hyperparams.assert_not_called()


def test_on_train_start_no_loggers_is_noop():
    """The hook does nothing when the loggers list is empty."""
    module = _make_module()
    module.trainer = SimpleNamespace(loggers=[])

    module.on_train_start()  # must not raise


def test_on_train_start_multiple_tb_loggers_each_receive_call():
    """When multiple TensorBoardLoggers are attached, each gets log_hyperparams."""
    module = _make_module()
    tb1 = MagicMock(spec=lightning.pytorch.loggers.TensorBoardLogger)
    tb2 = MagicMock(spec=lightning.pytorch.loggers.TensorBoardLogger)
    module.trainer = SimpleNamespace(loggers=[tb1, tb2])

    module.on_train_start()

    tb1.log_hyperparams.assert_called_once()
    tb2.log_hyperparams.assert_called_once()
