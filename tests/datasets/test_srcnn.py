from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from sisr.datasets.srcnn import TrainDataset, ValidationDataset


# ---------------------------------------------------------------------------
# TrainDataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_srcnn_train_ds(tmp_path_factory) -> TrainDataset:
    """One inline-built SRCNN train cache (subimg_size=20, stride=8) shared by
    every read-only test in this module.

    Module-scoped so the LMDB build runs once instead of once per test;
    build_num_workers=1 forces PR 1's inline (no ProcessPool) build path, which
    keeps it safe when the whole run is under pytest-xdist workers.
    """
    img_dir = tmp_path_factory.mktemp("shared_srcnn_hr")
    rng = np.random.default_rng(seed=0)
    for i in range(3):
        arr = rng.integers(0, 256, size=(36, 36, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i:02d}.png")
    return TrainDataset(
        img_dir=img_dir,
        subimg_size=20,
        stride=8,
        scale=2,
        blur_sigma=1.0,
        use_tqdm=False,
        cache_dir=img_dir / ".lmdb_cache_shared",
        build_num_workers=1,
    )


def _make_train(image_dir: Path, **overrides) -> TrainDataset:
    defaults = {
        "img_dir": image_dir,
        "subimg_size": 33,
        "stride": 14,
        "scale": 2,
        "blur_sigma": 1.0,
        "use_tqdm": False,
        "cache_dir": image_dir / ".lmdb_cache_train",
        "build_num_workers": 1,
    }
    defaults.update(overrides)
    return TrainDataset(**defaults)


def test_train_dataset_builds_and_len_positive(shared_srcnn_train_ds: TrainDataset):
    assert len(shared_srcnn_train_ds) > 0


def test_train_dataset_getitem_shape_dtype_range(shared_srcnn_train_ds: TrainDataset):
    lr, hr = shared_srcnn_train_ds[0]
    assert lr.shape == (3, 20, 20)
    assert hr.shape == (3, 20, 20)
    assert lr.dtype == torch.float32
    assert hr.dtype == torch.float32
    assert 0.0 <= lr.min() <= lr.max() <= 1.0
    assert 0.0 <= hr.min() <= hr.max() <= 1.0


def test_train_dataset_missing_key_raises_keyerror(shared_srcnn_train_ds: TrainDataset):
    """A missing LMDB key (here, an out-of-range index) surfaces as a KeyError
    naming the key, not a cryptic numpy TypeError from np.frombuffer(None)."""
    with pytest.raises(KeyError, match=r"lr_\d{8}"):
        shared_srcnn_train_ds[len(shared_srcnn_train_ds) + 100]


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


def test_train_dataset_checksum_includes_transforms_impl_tag(shared_srcnn_train_ds: TrainDataset):
    """The checksum input must include 'transforms_impl=albumentations' so caches
    built under the old PIL implementation invalidate automatically on upgrade."""
    import hashlib
    ds = shared_srcnn_train_ds
    file_manifest = ','.join(
        f'{p.name}:{p.stat().st_size}' for p in ds.img_paths
    )
    expected_canonical = '|'.join([
        file_manifest,
        '20',     # subimg_size
        '8',      # stride
        '2',      # scale
        '1.0',    # blur_sigma
        'transforms_impl=albumentations',
    ])
    expected = hashlib.sha256(expected_canonical.encode('utf-8')).hexdigest()
    assert ds._compute_checksum() == expected, (
        "checksum must include the 'transforms_impl=albumentations' tag — "
        "without it, caches built under the PIL implementation would be reused."
    )


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


def test_train_dataset_build_num_workers_1_builds_inline(tiny_rgb_image_dir: Path):
    """build_num_workers=1 must thread through to an inline LMDB build (no
    ProcessPoolExecutor), so the one-time build is safe inside a test/xdist
    worker instead of nesting an 8-process pool."""
    with patch("sisr.utils.ProcessPoolExecutor") as mock_pool:
        ds = _make_train(
            tiny_rgb_image_dir, subimg_size=20, stride=8, build_num_workers=1)
    mock_pool.assert_not_called()
    assert len(ds) > 0
    lr, hr = ds[0]
    assert lr.shape == (3, 20, 20)
    assert hr.shape == (3, 20, 20)


def test_patch_grid_derived_from_shared_helper(tiny_rgb_image_dir: Path):
    """_compute_offsets and the worker must derive the sliding-window grid from
    one shared helper, so their patch counts can never silently disagree and
    misalign the LMDB lr_/hr_ keys."""
    from sisr.datasets.srcnn import _iter_patch_origins, _process_subimages

    ds = _make_train(
        tiny_rgb_image_dir, subimg_size=20, stride=8, build_num_workers=1)
    _, total = ds._compute_offsets()

    # The worker's actual emitted pairs (one lr + one hr per patch) across every
    # image must sum to exactly the offset total.
    worker_total = 0
    for i, path in enumerate(ds.img_paths):
        pairs = _process_subimages(path, 20, 8, 2, 1.0, ds._img_offsets[i])
        worker_total += len(pairs) // 2
    assert worker_total == total

    # And the helper enumerates the expected grid for the 36x36 fixture:
    # crop 36 -> 36 (divisible by scale=2); (36-20)//8 + 1 = 3 positions per axis.
    origins = list(_iter_patch_origins(36, 36, 2, 20, 8))
    assert len(origins) == 9
    assert origins[0] == (0, 0)
    assert (16, 16) in origins


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


def test_validation_dataset_is_deterministic(tiny_rgb_image_dir: Path):
    """Calling __getitem__(idx) twice must return identical tensors —
    the validation pipeline has no random elements."""
    ds = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=2)
    lr_a, hr_a = ds[0]
    lr_b, hr_b = ds[0]
    assert torch.equal(lr_a, lr_b), "validation LR must be deterministic"
    assert torch.equal(hr_a, hr_b), "validation HR must be deterministic"


# ---------------------------------------------------------------------------
# SRDataset contract (P3.6 / P5.4)
# ---------------------------------------------------------------------------

def test_srcnn_datasets_are_srdataset_subclasses():
    from sisr.datasets.base import SRDataset
    assert issubclass(TrainDataset, SRDataset)
    assert issubclass(ValidationDataset, SRDataset)


def test_srcnn_validation_skips_non_image_files(tiny_rgb_image_dir: Path):
    """A stray .txt/extensionless file in the image dir must not enter
    img_paths — the old glob('*.*') would have matched the .txt."""
    (tiny_rgb_image_dir / "notes.txt").write_text("not an image")
    (tiny_rgb_image_dir / "MANIFEST").write_text("extensionless")
    ds = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=2)
    assert len(ds) == 3  # only the 3 PNGs
    assert all(p.suffix == ".png" for p in ds.img_paths)
