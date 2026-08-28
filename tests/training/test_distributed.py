"""Proof that the two-process test harness works, and what it can show.

Every distributed defect in this codebase is of one shape: something that must
happen once per *run* happens once per *process*. Catching that needs more than
one process, and nothing else in the suite starts one. These tests establish the
harness and pin the mechanism the rank-sensitive tests elsewhere depend on;
they assert nothing about ``sisr`` itself.
"""

import functools
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import lightning
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sisr.models.srcnn import SRCNN
from sisr.processors import RGBProcessor
from sisr.training import SRDataModule, SREvalConfig, SRLightning
from sisr.training.config import SRTrainingConfig

# Defined at module scope because ddp_spawn pickles the module across to each
# worker process; a locally-defined class cannot make that trip.


class _RankProbe(lightning.LightningModule):
    """Logs its own rank index twice, reduced and unreduced.

    With two processes the two logged values differ by construction: rank 0
    contributes 0.0 and rank 1 contributes 1.0. A reduced mean is therefore
    0.5 and an unreduced one is whatever the reporting process happened to
    hold, which makes the difference between the two impossible to miss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        (x,) = batch
        return self.layer(x).square().mean()

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        rank = torch.tensor(float(self.trainer.global_rank))
        self.log("probe/unreduced", rank, sync_dist=False)
        self.log("probe/reduced", rank, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _make_module() -> SRLightning:
    """A tiny real module -- the distributed behaviour under test is the project's."""
    return SRLightning(
        model=SRCNN(num_channels=3, num_filters=(8, 4), kernel_sizes=(3, 1, 3), padding="same"),
        processor=RGBProcessor(),
        training_config=SRTrainingConfig(scale=2),
        eval_config=SREvalConfig(crop_border=0),
        optimizer=functools.partial(torch.optim.SGD, lr=1e-4),
    )


def _make_datamodule(image_dir: Path, tmp_path: Path) -> SRDataModule:
    """Train plus a primary val set and one benchmark set, over the fixture images."""
    train = {
        "class_path": "sisr.datasets.srcnn.TrainDataset",
        "init_args": {
            "img_dir": str(image_dir),
            "subimg_size": 33,
            "stride": 33,
            "scale": 2,
            "use_tqdm": False,
            "cache_dir": str(tmp_path / ".lmdb_ddp"),
        },
    }
    val = {
        "class_path": "sisr.datasets.srcnn.ValidationDataset",
        "init_args": {"img_dir": str(image_dir), "scale": 2},
    }
    return SRDataModule(
        train_dataset=train,
        val_dataset=val,
        test_datasets={"Set5": val},
        train_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
        val_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
        test_dataloader_kwargs={"batch_size": 1, "num_workers": 0},
    )


def _loaders() -> tuple[DataLoader, DataLoader]:
    data = TensorDataset(torch.randn(8, 4, generator=torch.Generator().manual_seed(0)))
    return DataLoader(data, batch_size=2), DataLoader(data, batch_size=2)


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_harness_starts_two_real_processes(
    two_process_trainer: Callable[..., lightning.Trainer],
) -> None:
    """Fails if the harness silently degrades to a single process.

    A one-process run passes every rank assertion trivially, so a harness that
    quietly falls back would turn every test built on it green and meaningless.
    """
    trainer = two_process_trainer(max_steps=2, limit_val_batches=1)
    train_loader, val_loader = _loaders()
    trainer.fit(_RankProbe(), train_loader, val_loader)

    assert trainer.world_size == 2


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_sync_dist_reduces_across_processes_and_its_absence_does_not(
    two_process_trainer: Callable[..., lightning.Trainer],
) -> None:
    """Pins the mechanism every rank-correctness test in this suite relies on.

    ``sync_dist=True`` must produce the mean over both processes; without it the
    logged value is one process's own, presented as though it were global. That
    second case is the live defect shape for validation metrics, so it is
    asserted rather than merely described.
    """
    trainer = two_process_trainer(max_steps=2, limit_val_batches=1)
    train_loader, val_loader = _loaders()
    trainer.fit(_RankProbe(), train_loader, val_loader)

    metrics = {key: float(value) for key, value in trainer.callback_metrics.items()}
    # mean(0.0, 1.0) -- only reachable if both processes contributed.
    assert metrics["probe/reduced"] == pytest.approx(0.5)
    # The reporting process's own rank index, never the mean.
    assert metrics["probe/unreduced"] == pytest.approx(0.0)


class _CountsWhatEachProcessSaw(lightning.Callback):
    """Records per-process batch counts to disk, since ddp_spawn discards workers.

    Written to a rank-named file rather than returned: the spawned processes are
    gone by the time the test resumes, so anything they learned has to survive
    them.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.train = 0
        self.val = 0

    def on_train_batch_end(self, trainer, pl_module, *args) -> None:
        self.train += 1

    def on_validation_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        self.val += 1

    def on_fit_end(self, trainer, pl_module) -> None:
        path = self.out_dir / f"rank{trainer.global_rank}.json"
        path.write_text(json.dumps({"train": self.train, "val": self.val}))


@pytest.mark.filterwarnings("ignore::lightning.pytorch.utilities.warnings.PossibleUserWarning")
def test_training_is_split_across_processes_and_evaluation_is_not(
    tiny_rgb_image_dir: Path, tmp_path: Path, two_process_trainer
) -> None:
    """The invariant the whole distributed design rests on.

    Training splits, because that is what distributing it means. Evaluation does
    **not**, because ``DistributedSampler`` pads a set to a multiple of the world
    size by repeating samples -- a five-image benchmark across four processes
    would report a figure for eight images with three counted twice, which is not
    that set's score. Every process therefore scores every image.
    """
    counts_dir = tmp_path / "counts"
    counts_dir.mkdir()
    module = _make_module()
    datamodule = _make_datamodule(tiny_rgb_image_dir, tmp_path)

    trainer = two_process_trainer(
        max_epochs=1,
        limit_train_batches=1.0,
        use_distributed_sampler=False,
        callbacks=[_CountsWhatEachProcessSaw(counts_dir)],
    )
    trainer.fit(module, datamodule=datamodule)

    counts = [json.loads((counts_dir / f"rank{r}.json").read_text()) for r in (0, 1)]
    # Three training samples split evenly over two processes -- and note the
    # total is FOUR, not three: DistributedSampler pads by repeating one sample
    # so the shards are equal. Harmless while training, and precisely why the
    # evaluation loaders below must never go through it.
    assert [c["train"] for c in counts] == [2, 2], counts
    # Both loaders in the val list, whole, on both processes: 3 images each.
    assert all(c["val"] == 6 for c in counts), counts


def test_a_distributed_run_refuses_lightning_s_own_sampler_injection() -> None:
    """Leaving the injection on splits the benchmark sets, silently and on the
    metric path -- so it is refused rather than warned about."""
    module = _make_module()
    module.trainer = SimpleNamespace(
        world_size=2, _accelerator_connector=SimpleNamespace(use_distributed_sampler=True)
    )

    with pytest.raises(RuntimeError, match="use_distributed_sampler"):
        module.on_fit_start()


def test_a_single_process_run_is_unaffected_by_the_check() -> None:
    module = _make_module()
    module.trainer = SimpleNamespace(
        world_size=1, _accelerator_connector=SimpleNamespace(use_distributed_sampler=True)
    )

    module.on_fit_start()  # must not raise
