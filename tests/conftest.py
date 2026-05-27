"""Shared pytest fixtures for the sisr test suite."""
from pathlib import Path

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
