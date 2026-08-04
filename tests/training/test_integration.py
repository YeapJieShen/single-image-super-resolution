"""End-to-end Trainer integration for the SR training stack.

A single ``Trainer(fast_dev_run=True)`` fit+test over the tiny fixture images
exercises the wiring the per-method unit tests in ``test_lightning_module.py``
cannot reach: ``training_step`` / ``validation_step`` under a real loop,
``dataloader_idx`` routing across the ``[primary_val, Set5]`` val-loader list,
the metric names actually logged, and the module -> datamodule ->
``BenchmarkImageLogger`` integration (the callback auto-discovers ``Set5`` from
``datamodule.test_names``). It is also the suite's canary for the lightning
``LeafSpec`` pytree ``FutureWarning``: only a real Trainer loop emits it,
and the strict ``filterwarnings=error`` config turns it into a failure unless
the scoped ignore in ``pyproject.toml`` is in place.
"""

import functools
from pathlib import Path

import lightning
import pytest
import torch
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from torch.utils.data import DataLoader, Dataset

from sisr.models.srcnn import SRCNN
from sisr.models.srresnet import SRResNetTrainingConfig
from sisr.models.srresnet.model import SRResNet
from sisr.processors import RGBProcessor, YChannelProcessor
from sisr.training import (
    BenchmarkImageLogger,
    SRDataModule,
    SREvalConfig,
    SRLightning,
    SRPredictionWriter,
    SRTrainingConfig,
)
from sisr.training.cuda_graph import CUDAGraphStep


def _make_srcnn_module() -> SRLightning:
    """SRLightning wrapping a 3-channel SRCNN with the RGB pass-through processor."""
    model = SRCNN(num_channels=3, num_filters=(64, 32), kernel_sizes=(9, 1, 5), padding=0)
    return SRLightning(
        model=model,
        processor=RGBProcessor(),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def _make_datamodule(image_dir: Path) -> SRDataModule:
    """SRDataModule over one tiny image dir: train + primary val + one Set5 test set."""
    train_spec = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(image_dir),
            "subimg_size": 33,
            "stride": 14,
            "scale": 2,
            "use_tqdm": False,
            "cache_dir": str(image_dir / ".lmdb_cache_integration"),
        },
    }
    val_spec = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(image_dir), "scale": 2},
    }
    return SRDataModule(
        train_dataset=train_spec,
        val_dataset=val_spec,
        test_datasets={"Set5": val_spec},
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
        test_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


# num_workers=0 (tiny fixture) and accelerator="cpu" on a CUDA box both raise
# PossibleUserWarnings — environment noise, not a library-health signal — so
# silence them for this test only, leaving the strict global filter honest elsewhere.
@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_fast_dev_run_fit_and_test_logs_module_and_callback_metrics(
    tiny_rgb_image_dir: Path,
):
    """Fails if a real fit+test loop stops logging the expected metric keys.

    Catches wiring regressions the unit tests can't: a broken ``training_step`` /
    ``validation_step``, mis-routed ``dataloader_idx``, renamed metric keys, or a
    ``BenchmarkImageLogger`` that no longer discovers the datamodule's test set.
    ``accelerator="cpu"`` keeps CI (no GPU) and local (GPU) runs identical.
    """
    module = _make_srcnn_module()
    datamodule = _make_datamodule(tiny_rgb_image_dir)
    trainer = lightning.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[BenchmarkImageLogger(log_every_n_val_runs=1)],
    )

    trainer.fit(module, datamodule=datamodule)
    fit_metrics = set(trainer.callback_metrics)
    # Module-logged: training_step + validation_step on the primary val loader (idx 0).
    # ssim/val/{RGB,Y} — eval_config.ssim_channels defaults to ['RGB', 'Y'].
    assert {"loss/train", "loss/val", "psnr/val/RGB", "ssim/val/RGB", "ssim/val/Y"} <= fit_metrics
    # Callback-logged for the Set5 test loader (val_dataloader idx 1) — proves
    # BenchmarkImageLogger auto-discovered the datamodule's test set and ran.
    assert {"psnr/Set5/RGB", "ssim/Set5/RGB", "ssim/Set5/Y"} <= fit_metrics

    trainer.test(module, datamodule=datamodule)
    test_metrics = set(trainer.callback_metrics)
    # test_step is a no-op; these come solely from BenchmarkImageLogger.on_test_*.
    assert {"psnr/Set5/RGB", "ssim/Set5/RGB", "ssim/Set5/Y"} <= test_metrics


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_fast_dev_run_raises_on_cross_wired_model_and_dataset_through_real_trainer(
    tiny_rgb_image_dir: Path,
):
    """Regression: SRLightning.setup()'s input_contract probe must actually
    fire through a real Trainer.fit() call, not only the SimpleNamespace-
    stubbed trainer every setup() unit test in test_lightning_module.py uses.
    Locks Lightning's documented DataModule.setup-before-LightningModule.setup
    hook ordering (Trainer._call_setup_hook) against a silent regression that
    every stubbed-trainer test would stay green through.
    """
    model = SRResNet(scale=2, num_residual_blocks=1)
    module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        training_config=SRResNetTrainingConfig(scale=2),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    # srcnn's datasets are pre_upsampled (lr.shape == hr.shape) -- mismatched
    # against SRResNet's native_lr contract.
    datamodule = _make_datamodule(tiny_rgb_image_dir)
    trainer = lightning.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with pytest.raises(ValueError, match="input_contract"):
        trainer.fit(module, datamodule=datamodule)


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_fast_dev_run_reconstruct_never_runs_mid_training_step(tiny_rgb_image_dir: Path):
    """training_step passes need_sr_rgb=False (see SRLightning._forward_lr), so
    processor.reconstruct must never fire while the module is in train mode —
    only validation_step (self.training False) may call it. A real Trainer
    loop, not a direct _step() call, is the only way to prove training_step
    itself (not just _step) actually requests the skip."""
    module = _make_srcnn_module()
    datamodule = _make_datamodule(tiny_rgb_image_dir)

    call_training_flags = []
    real_reconstruct = module.processor.reconstruct

    def _spy(*args, **kwargs):
        call_training_flags.append(module.training)
        return real_reconstruct(*args, **kwargs)

    module.processor.reconstruct = _spy

    trainer = lightning.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    assert call_training_flags, "expected at least one reconstruct call, from validation_step"
    assert not any(call_training_flags), "reconstruct must never run while self.training is True"


# ---------------------------------------------------------------------------
# cli predict end-to-end — LR-only inference path, both architectures
# ---------------------------------------------------------------------------


def _make_predict_datamodule(image_dir: Path, arch: str) -> SRDataModule:
    """Predict-only SRDataModule: train_dataset/val_dataset are required by
    the constructor but never instantiated — setup(stage='predict') only
    builds predict_dataset, matching every other stage's selective build."""
    unused_spec = {
        "class_path": f"sisr.datasets.{arch}.ValidationDataset",
        "init_args": {"img_dir": str(image_dir), "scale": 2},
    }
    return SRDataModule(
        train_dataset=unused_spec,
        val_dataset=unused_spec,
        predict_dataset={
            "class_path": "sisr.datasets.predict.PredictDataset",
            "init_args": {"img_dir": str(image_dir)},
        },
        predict_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_predict_end_to_end_srcnn_y_channel_same_size_output(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """SRCNN's pre-upsampled, Y-channel path: predict output must be the same
    H/W as the input (scale=1x) and land as one PNG per input file."""
    from PIL import Image

    model = SRCNN(num_channels=1, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same")
    module = SRLightning(
        model=model,
        processor=YChannelProcessor(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    datamodule = _make_predict_datamodule(tiny_rgb_image_dir, arch="srcnn")
    out_dir = tmp_path / "predictions"
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[SRPredictionWriter(output_dir=out_dir)],
    )

    predictions = trainer.predict(module, datamodule=datamodule)

    assert len(predictions) == 3  # tiny_rgb_image_dir has 3 images, batch_size=1
    for pred in predictions:
        assert pred.shape == (1, 3, 36, 36)  # 36x36 input, scale=1x

    written = sorted(p.stem for p in out_dir.glob("*.png"))
    assert written == ["img_00", "img_01", "img_02"]
    for name in written:
        with Image.open(out_dir / f"{name}.png") as img:
            assert img.size == (36, 36)


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_predict_end_to_end_srresnet_rgb_upsamples_by_scale(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """SRResNet's genuine-LR, RGB path: predict output must be exactly
    scale x the input H/W."""
    from PIL import Image

    model = SRResNet(scale=2, num_residual_blocks=1)
    module = SRLightning(
        model=model,
        processor=RGBProcessor(),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )
    datamodule = _make_predict_datamodule(tiny_rgb_image_dir, arch="srresnet")
    out_dir = tmp_path / "predictions"
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[SRPredictionWriter(output_dir=out_dir)],
    )

    predictions = trainer.predict(module, datamodule=datamodule)

    assert len(predictions) == 3
    for pred in predictions:
        assert pred.shape == (1, 3, 72, 72)  # 36x36 input, scale=2x -> 72x72

    written = sorted(p.stem for p in out_dir.glob("*.png"))
    assert written == ["img_00", "img_01", "img_02"]
    for name in written:
        with Image.open(out_dir / f"{name}.png") as img:
            assert img.size == (72, 72)


# ---------------------------------------------------------------------------
# cuda_graph — graphed training must be indistinguishable from eager
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs a CUDA device"
)


class _FixedPairs(Dataset):
    """Deterministic same-shape ``(lr, hr)`` RGB pairs — no decode, no randomness."""

    def __init__(self, n: int):
        g = torch.Generator().manual_seed(0)
        self.pairs = [
            (torch.rand(3, 16, 16, generator=g), torch.rand(3, 16, 16, generator=g))
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[i]


class _LossTrace(lightning.Callback):
    """Records the per-step training loss so two runs can be compared exactly."""

    def __init__(self):
        self.losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.losses.append(float(outputs["loss"]))


def _run_graph_fit(
    cuda_graph: bool, n_samples: int, max_steps: int, lr_scheduler=None
) -> tuple[list[float], SRLightning]:
    """Fit a graphed-or-eager SRCNN over ``_FixedPairs`` and return its loss trace."""
    lightning.seed_everything(7, verbose=False)
    module = SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(cuda_graph=cuda_graph),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-3, momentum=0.9),
        lr_scheduler=lr_scheduler,
    )
    trace = _LossTrace()
    trainer = lightning.Trainer(
        accelerator="cuda",
        devices=1,
        max_steps=max_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        benchmark=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[trace],
    )
    loader = DataLoader(_FixedPairs(n_samples), batch_size=4, num_workers=0, shuffle=False)
    trainer.fit(module, train_dataloaders=loader)
    return trace.losses, module


@requires_cuda
@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_cuda_graph_loss_trace_is_bit_identical_to_eager():
    """Graphed training must be numerically indistinguishable from eager, not
    merely close: the capture warm-up runs three extra backward passes, and any
    of them leaking into the weights (or a missed zero_grad) shifts the trace."""
    eager, _ = _run_graph_fit(cuda_graph=False, n_samples=16, max_steps=12)
    graphed, module = _run_graph_fit(cuda_graph=True, n_samples=16, max_steps=12)

    assert module._cuda_graph is not None and module._cuda_graph.captured
    assert len(graphed) == 12
    assert graphed == eager


@requires_cuda
@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_cuda_graph_partial_last_batch_falls_back_without_stale_gradients():
    """14 samples at batch 4 ends every epoch with a 2-sample batch the graph
    cannot replay. Gating the backward/zero_grad no-ops per run instead of per
    step makes that batch step on the previous replay's gradients — a silent
    duplicated update this exact-equality trace catches."""
    eager, _ = _run_graph_fit(cuda_graph=False, n_samples=14, max_steps=12)
    graphed, _ = _run_graph_fit(cuda_graph=True, n_samples=14, max_steps=12)

    assert graphed == eager


@requires_cuda
@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_cuda_graph_leaves_lr_scheduler_effective():
    """optimizer.step() is deliberately left out of the capture, so a scheduler's
    learning-rate changes still reach the weights. Capturing it would bake the LR
    in as a graph constant (torch.optim.SGD has no `capturable` opt-out) and
    every scheduler would silently no-op."""
    scheduler = functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5)
    eager, _ = _run_graph_fit(False, n_samples=16, max_steps=12, lr_scheduler=scheduler)
    graphed, module = _run_graph_fit(True, n_samples=16, max_steps=12, lr_scheduler=scheduler)

    assert graphed == eager
    assert module.optimizers().param_groups[0]["lr"] < 1e-3


def _graph_srcnn_module() -> SRLightning:
    """The module `_run_graph_fit` builds, for tests that need to reuse one."""
    return SRLightning(
        model=SRCNN(num_channels=3, num_filters=(4, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(cuda_graph=True),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-3, momentum=0.9),
    )


def _graph_trainer(callbacks: list, **overrides) -> lightning.Trainer:
    return lightning.Trainer(
        accelerator="cuda",
        devices=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        benchmark=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=callbacks,
        **overrides,
    )


@requires_cuda
@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
@pytest.mark.filterwarnings("ignore:When using:UserWarning")
def test_cuda_graph_refuses_accumulation_scheduled_mid_run():
    """GradientAccumulationScheduler only raises accumulate_grad_batches from its
    own on_train_epoch_start, so a refusal checked once at fit start is
    guaranteed to miss it and training would silently degrade from the scheduled
    epoch — every replay re-zeroes, leaving only the last micro-batch's
    gradient, unscaled. Lightning merely warns about our optimizer_zero_grad
    override here; it does not stop the run."""
    lightning.seed_everything(7, verbose=False)
    module = _graph_srcnn_module()
    trainer = _graph_trainer(
        [GradientAccumulationScheduler(scheduling={2: 2})], max_epochs=3, max_steps=-1
    )
    loader = DataLoader(_FixedPairs(16), batch_size=4, num_workers=0, shuffle=False)

    with pytest.raises(RuntimeError, match="does not support gradient accumulation"):
        trainer.fit(module, train_dataloaders=loader)


@requires_cuda
@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_cuda_graph_recaptures_on_a_second_fit():
    """Strategy.teardown moves the module and its .grad tensors back to CPU at the
    end of a fit, freeing the blocks the graph baked addresses for. A second fit
    must capture afresh; replaying the first fit's graph writes into freed
    memory."""
    lightning.seed_everything(7, verbose=False)
    module = _graph_srcnn_module()
    loader = DataLoader(_FixedPairs(16), batch_size=4, num_workers=0, shuffle=False)

    graph_ids = []
    for _ in range(2):
        trace = _LossTrace()
        _graph_trainer([trace], max_steps=8).fit(module, train_dataloaders=loader)
        assert module._cuda_graph is not None and module._cuda_graph.captured
        graph_ids.append(id(module._cuda_graph))
        assert all(loss == loss for loss in trace.losses)  # no NaN from freed memory
        assert len(trace.losses) == 8

    assert graph_ids[0] != graph_ids[1], "second fit reused the first fit's dead graph"


@requires_cuda
def test_cuda_graph_capture_leaves_batchnorm_buffers_untouched():
    """The warm-up's forwards advance BatchNorm running stats, and SRResNet's
    residual blocks have BatchNorm — so capture would silently perturb them
    (decaying as 0.9^n) without the snapshot/restore. SRCNN has no buffers, so
    no SRCNN parity test can catch this."""
    lightning.seed_everything(7, verbose=False)
    module = SRLightning(
        model=SRResNet(scale=2, num_residual_blocks=1),
        processor=RGBProcessor(),
        training_config=SRResNetTrainingConfig(scale=2, cuda_graph=True),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-3),
    ).to("cuda")
    module.train()
    before = {name: buf.detach().clone() for name, buf in module.named_buffers()}
    assert any("running_mean" in name for name in before)

    step = CUDAGraphStep(
        lambda b: module._step(b, need_sr_rgb=False)[0], module, module.configure_optimizers()
    )
    step.capture((torch.rand(2, 3, 8, 8, device="cuda"), torch.rand(2, 3, 16, 16, device="cuda")))

    assert step.captured
    for name, buf in module.named_buffers():
        assert torch.equal(buf, before[name]), name
