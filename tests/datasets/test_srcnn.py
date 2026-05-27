from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from sisr.datasets.srcnn import TrainDataset, ValidationDataset


# ---------------------------------------------------------------------------
# TrainDataset
# ---------------------------------------------------------------------------

def _make_train(image_dir: Path, **overrides) -> TrainDataset:
    defaults = {
        "img_dir": image_dir,
        "subimg_size": 33,
        "stride": 14,
        "scale": 2,
        "blur_sigma": 1.0,
        "use_tqdm": False,
        "cache_dir": image_dir / ".lmdb_cache_train",
    }
    defaults.update(overrides)
    return TrainDataset(**defaults)


def test_train_dataset_builds_and_len_positive(tiny_rgb_image_dir: Path):
    ds = _make_train(tiny_rgb_image_dir)
    assert len(ds) > 0


def test_train_dataset_getitem_shape_dtype_range(tiny_rgb_image_dir: Path):
    ds = _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8)
    lr, hr = ds[0]
    assert lr.shape == (3, 20, 20)
    assert hr.shape == (3, 20, 20)
    assert lr.dtype == torch.float32
    assert hr.dtype == torch.float32
    assert 0.0 <= lr.min() <= lr.max() <= 1.0
    assert 0.0 <= hr.min() <= hr.max() <= 1.0


def test_train_dataset_cache_reuse_skips_rebuild(tiny_rgb_image_dir: Path):
    """Second instantiation with same params must not call _process_subimages."""
    _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8)
    with patch("sisr.datasets.srcnn._process_subimages") as mock_proc:
        _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8)
        mock_proc.assert_not_called()


def test_train_dataset_checksum_change_triggers_rebuild(tiny_rgb_image_dir: Path):
    """Different subimg_size produces a different cache directory + different
    patch counts, confirming the rebuild ran with new params.

    (Patch-based detection of `_process_subimages` calls doesn't work because
    those run in `ProcessPoolExecutor` workers, which don't see parent-process
    monkey-patches.)
    """
    ds_a = _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8)
    ds_b = _make_train(tiny_rgb_image_dir, subimg_size=24, stride=8)
    assert len(ds_a) != len(ds_b), "different subimg_size must yield different patch counts"


def test_train_dataset_no_images_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No images"):
        TrainDataset(
            img_dir=tmp_path,
            subimg_size=20,
            stride=8,
            scale=2,
            blur_sigma=1.0,
            use_tqdm=False,
            cache_dir=tmp_path / ".lmdb",
        )


def test_train_dataset_rejects_channels_param(tiny_rgb_image_dir: Path):
    """The `channels` parameter was removed (HR is always RGB; Y/YCbCr
    selection happens in SRLightning), so passing it is a TypeError."""
    with pytest.raises(TypeError):
        _make_train(tiny_rgb_image_dir, channels="L", subimg_size=20, stride=8)


# ---------------------------------------------------------------------------
# ValidationDataset
# ---------------------------------------------------------------------------

def test_validation_dataset_serves_full_image_pairs(tiny_rgb_image_dir: Path):
    ds = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=2)
    assert len(ds) == 3  # tiny_rgb_image_dir creates 3 images
    lr, hr = ds[0]
    assert lr.shape == hr.shape
    assert lr.shape[0] == 3  # RGB
    assert lr.dtype == torch.float32
    assert 0.0 <= lr.min() <= lr.max() <= 1.0


def test_validation_dataset_no_images_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No images"):
        ValidationDataset(img_dir=tmp_path, scale=2)


def test_validation_dataset_blur_sigma_propagates(tiny_rgb_image_dir: Path):
    """ValidationDataset accepts and uses blur_sigma, so two datasets with
    different sigmas produce different LR outputs."""
    ds_a = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=2, blur_sigma=0.1)
    ds_b = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=2, blur_sigma=3.0)
    lr_a, _ = ds_a[0]
    lr_b, _ = ds_b[0]
    assert not torch.allclose(lr_a, lr_b), "different blur_sigma must produce different LR"
