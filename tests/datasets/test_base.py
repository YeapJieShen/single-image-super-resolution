from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sisr.datasets.base import SRDataset


def _write_png(path: Path, size: int = 8) -> None:
    rng = np.random.default_rng(seed=abs(hash(path.name)) % (2**32))
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


class _Mini(SRDataset):
    """Minimal concrete SRDataset used to exercise the shared base helpers."""

    def __init__(self, img_dir):
        super().__init__()
        self._index_images(img_dir)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.img_paths[idx]


def test_srdataset_subclass_missing_abstract_methods_fails_at_construction():
    """A mis-shaped dataset (no __len__/__getitem__) must fail clearly at
    construction time via the ABC, not silently at first runtime access."""

    class Bad(SRDataset):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Bad()


def test_index_images_populates_img_paths_as_paths(tmp_path: Path):
    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "b.png")
    ds = _Mini(tmp_path)
    assert [p.name for p in ds.img_paths] == ["a.png", "b.png"]  # sorted
    assert all(isinstance(p, Path) for p in ds.img_paths)
    assert ds.img_dir == tmp_path


def test_index_images_skips_non_image_and_extensionless_files(tmp_path: Path):
    """P5.4: the allowlist keeps only real image extensions — a .txt file and
    an extensionless file are excluded (the old glob('*.*') matched the .txt
    and dropped the extensionless one silently)."""
    _write_png(tmp_path / "img_00.png")
    _write_png(tmp_path / "img_01.png")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "README").write_text("extensionless")
    (tmp_path / "checksum.json").write_text("{}")
    ds = _Mini(tmp_path)
    assert [p.name for p in ds.img_paths] == ["img_00.png", "img_01.png"]


def test_index_images_no_images_raises(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("not an image")
    with pytest.raises(ValueError, match="No images"):
        _Mini(tmp_path)


def test_load_rgb_returns_hwc_uint8_rgb(tmp_path: Path):
    _write_png(tmp_path / "a.png", size=8)
    arr = SRDataset._load_rgb(tmp_path / "a.png")
    assert arr.shape == (8, 8, 3)
    assert arr.dtype == np.uint8


def test_to_tensor_produces_chw_float01(tmp_path: Path):
    arr = np.full((4, 4, 3), 255, dtype=np.uint8)
    t = SRDataset._to_tensor(arr)
    assert t.shape == (3, 4, 4)
    assert t.dtype == torch.float32
    assert torch.allclose(t, torch.ones(3, 4, 4))
