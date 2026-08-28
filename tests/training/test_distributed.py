"""Proof that the two-process test harness works, and what it can show.

Every distributed defect in this codebase is of one shape: something that must
happen once per *run* happens once per *process*. Catching that needs more than
one process, and nothing else in the suite starts one. These tests establish the
harness and pin the mechanism the rank-sensitive tests elsewhere depend on;
they assert nothing about ``sisr`` itself.
"""

from collections.abc import Callable

import lightning
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

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
