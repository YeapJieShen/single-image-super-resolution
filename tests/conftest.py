"""Shared pytest fixtures for the sisr test suite."""

from collections.abc import Callable
from pathlib import Path

import lightning
import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tiny_rgb_image_dir(tmp_path: Path) -> Path:
    """Tmp dir with 3 small RGB PNGs.

    Sized 36x36 so they're cleanly divisible by scale=2/3/4 and large enough
    for 33x33 sub-image extraction by SRCNN's TrainDataset.
    """
    rng = np.random.default_rng(seed=0)
    for i in range(3):
        arr = rng.integers(0, 256, size=(36, 36, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"img_{i:02d}.png")
    return tmp_path


@pytest.fixture
def rgb_batch() -> torch.Tensor:
    """Random ``(B=2, C=3, H=8, W=8)`` RGB tensor in ``[0, 1]``."""
    return torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def two_process_trainer() -> Callable[..., lightning.Trainer]:
    """Factory for a real two-process trainer, over gloo, on CPU.

    Nothing else in the suite runs more than one process, so without this every
    rank-sensitive code path ships unproven. Two processes is enough: the whole
    class of defect here is "this happens once per process when it should happen
    once", and that is already visible at a world size of two.

    ``ddp_spawn`` rather than ``ddp`` because ``ddp`` re-executes the entry
    script, which under pytest means re-running the whole test command. The
    launcher differs from production; the semantics under test — sampler
    splitting, metric reduction, rank identity — are the strategy's, and are the
    same either way.

    **What this cannot prove:** anything NCCL-specific, device placement on real
    GPUs, or throughput. Those need genuine multi-GPU hardware, which containers
    do not supply — they share the host's devices.

    Returns:
        A callable taking ``Trainer`` keyword overrides and returning a trainer
        configured for two CPU processes.
    """

    def _make(**overrides: object) -> lightning.Trainer:
        settings: dict[str, object] = {
            "accelerator": "cpu",
            "devices": 2,
            "strategy": "ddp_spawn",
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        }
        settings.update(overrides)
        return lightning.Trainer(**settings)  # type: ignore[arg-type]

    return _make
