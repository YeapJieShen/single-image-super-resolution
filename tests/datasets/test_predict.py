from pathlib import Path

import pytest
import torch

from sisr.datasets.base import SRDataset
from sisr.datasets.predict import PredictDataset


def test_predict_dataset_returns_lr_tensor_only(tiny_rgb_image_dir: Path):
    """__getitem__ returns a bare tensor (no HR pair) — the LR-only contract."""
    ds = PredictDataset(img_dir=tiny_rgb_image_dir)
    assert len(ds) == 3
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (3, 36, 36)
    assert item.dtype == torch.float32
    assert 0.0 <= item.min() <= item.max() <= 1.0


def test_predict_dataset_no_images_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No images"):
        PredictDataset(img_dir=tmp_path)


def test_predict_dataset_duplicate_stems_raise(tmp_path: Path):
    """Outputs are named by stem, so `cat.png` + `cat.jpg` would collide."""
    import numpy as np
    from PIL import Image

    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "cat.png")
    Image.fromarray(arr).save(tmp_path / "cat.jpg")

    with pytest.raises(ValueError, match="Duplicate filename stem"):
        PredictDataset(img_dir=tmp_path)


def test_predict_dataset_is_srdataset_subclass():
    assert issubclass(PredictDataset, SRDataset)


def test_predict_dataset_skips_non_image_files(tiny_rgb_image_dir: Path):
    (tiny_rgb_image_dir / "notes.txt").write_text("not an image")
    (tiny_rgb_image_dir / "MANIFEST").write_text("extensionless")
    ds = PredictDataset(img_dir=tiny_rgb_image_dir)
    assert len(ds) == 3
    assert all(p.suffix == ".png" for p in ds.img_paths)


def test_predict_dataset_is_deterministic(tiny_rgb_image_dir: Path):
    """No random augmentation — real inference input must round-trip identically."""
    ds = PredictDataset(img_dir=tiny_rgb_image_dir)
    a = ds[0]
    b = ds[0]
    assert torch.equal(a, b)


def test_predict_dataset_does_not_resize_or_downsample(tiny_rgb_image_dir: Path):
    """Unlike ValidationDataset, PredictDataset must not synthesize LR from HR —
    it serves genuine LR images exactly as found on disk."""
    from PIL import Image

    path = sorted(tiny_rgb_image_dir.glob("*.png"))[0]
    expected = torch.from_numpy(SRDataset._load_rgb(path)).permute(2, 0, 1).float().div(255.0)
    ds = PredictDataset(img_dir=tiny_rgb_image_dir)
    idx = ds.img_paths.index(path)
    torch.testing.assert_close(ds[idx], expected)
    Image.open(path).close()  # sanity: file still readable, unmodified
