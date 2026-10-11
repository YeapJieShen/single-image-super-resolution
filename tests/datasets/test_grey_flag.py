"""Eval datasets read the file mode and yield it only when asked (grey-plane scoring)."""

import numpy as np
import pytest
import torch
from PIL import Image

from sisr.datasets import native_lr, pre_upsampled

DATASETS = [native_lr.ValidationDataset, pre_upsampled.ValidationDataset]


@pytest.fixture
def img_dir(tmp_path):
    rng = np.random.default_rng(0)
    g = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    Image.fromarray(g, "L").save(tmp_path / "a_grey_L.png")  # single channel on disk
    Image.fromarray(np.stack([g] * 3, -1)).save(tmp_path / "b_equal_rgb.png")  # man.png-style
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(tmp_path / "c_col.png")
    return tmp_path


@pytest.mark.parametrize("cls", DATASETS)
def test_off_yields_the_two_tuple_it_always_did(cls, img_dir):
    item = cls(img_dir, 2)[0]
    assert len(item) == 2


@pytest.mark.parametrize("cls", DATASETS)
def test_on_flags_a_single_channel_file_and_nothing_else(cls, img_dir):
    ds = cls(img_dir, 2, yield_is_grey=True)
    flags = {p.name: bool(ds[i][2]) for i, p in enumerate(ds.img_paths)}

    assert flags == {"a_grey_L.png": True, "b_equal_rgb.png": False, "c_col.png": False}
    lr, hr, grey = ds[0]
    assert isinstance(grey, torch.Tensor) and grey.dtype == torch.bool and grey.ndim == 0


@pytest.mark.parametrize("cls", DATASETS)
def test_the_flag_batches_to_one_bool_per_image(cls, img_dir):
    ds = cls(img_dir, 2, yield_is_grey=True)
    _, _, grey = next(iter(torch.utils.data.DataLoader(ds, batch_size=3)))
    assert grey.tolist() == [True, False, False]
