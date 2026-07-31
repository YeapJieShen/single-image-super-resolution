from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from sisr.datasets.srresnet import TrainDataset, ValidationDataset


@pytest.fixture
def varied_size_rgb_image_dir(tmp_path: Path) -> Path:
    """Tmp dir with 3 RGB PNGs of different, non-square sizes.

    Exercises the per-image ``(H, W)`` shape header the LMDB cache stores —
    DIV2K images vary in size, unlike SRCNN's fixed sub-image constants.
    """
    rng = np.random.default_rng(seed=1)
    for i, (h, w) in enumerate([(36, 36), (48, 32), (30, 50)]):
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"img_{i:02d}.png")
    return tmp_path


# ---------------------------------------------------------------------------
# TrainDataset (random-crop)
# ---------------------------------------------------------------------------


def test_train_dataset_getitem_lr_is_downscaled_hr(tiny_rgb_image_dir: Path):
    """LR must be the cv2 INTER_CUBIC downscale of the SAME HR crop the item
    returned — not merely a (3, h/scale, w/scale) tensor in [0, 1]. Recovers the
    returned HR crop as uint8 and re-runs the dataset's own LR pipeline as the
    reference; the crop is random, but LR and HR come from one call, so the
    reference is exact. A regression to bilinear/nearest downscaling fails here."""
    import albumentations as A
    import cv2
    import numpy as np
    from albumentations.pytorch import ToTensorV2

    ds = TrainDataset(img_dir=tiny_rgb_image_dir, scale=4, hr_crop_size=16)
    lr, hr = ds[0]

    # Shape / dtype / range (retained smoke coverage).
    assert hr.shape == (3, 16, 16)
    assert lr.shape == (3, 4, 4)  # 16 // 4
    assert lr.dtype == torch.float32 and hr.dtype == torch.float32
    assert 0.0 <= lr.min() <= lr.max() <= 1.0
    assert 0.0 <= hr.min() <= hr.max() <= 1.0

    # Reference: recover the HR crop as uint8 HWC and re-run the dataset's own
    # LR pipeline (A.Resize INTER_CUBIC -> ToFloat -> ToTensorV2).
    hr_uint8 = (hr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    lr_size = 16 // 4
    ref_pipeline = A.Compose(
        [
            A.Resize(lr_size, lr_size, interpolation=cv2.INTER_CUBIC),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    )
    lr_expected = ref_pipeline(image=hr_uint8)["image"]
    torch.testing.assert_close(lr, lr_expected, atol=1e-6, rtol=0)


def test_train_dataset_len_scales_with_crops_per_image(tiny_rgb_image_dir: Path):
    ds = TrainDataset(img_dir=tiny_rgb_image_dir, scale=2, hr_crop_size=16, crops_per_image=4)
    assert len(ds) == 3 * 4  # tiny_rgb_image_dir has 3 images


def test_train_dataset_crop_size_not_divisible_by_scale_raises(tiny_rgb_image_dir: Path):
    with pytest.raises(ValueError, match="divisible"):
        TrainDataset(img_dir=tiny_rgb_image_dir, scale=4, hr_crop_size=18)


def test_train_dataset_crop_larger_than_image_raises(tiny_rgb_image_dir: Path):
    # Images are 36x36; a 40px crop cannot fit.
    ds = TrainDataset(img_dir=tiny_rgb_image_dir, scale=2, hr_crop_size=40)
    with pytest.raises(ValueError, match="smaller than hr_crop_size"):
        ds[0]


def test_train_dataset_no_images_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No images"):
        TrainDataset(img_dir=tmp_path, scale=2, hr_crop_size=16)


def test_train_dataset_random_crop_varies_across_calls(tiny_rgb_image_dir: Path):
    """A.RandomCrop should produce different crops across calls on a 36x36
    image with crop_size=16 (many valid (top, left) positions). With 8
    sampled calls the probability of all-identical is ~(1/441)**7, far below
    a reasonable flake threshold."""
    ds = TrainDataset(img_dir=tiny_rgb_image_dir, scale=4, hr_crop_size=16)
    samples = [ds[0] for _ in range(8)]
    hrs = [hr for _, hr in samples]
    differ = any(
        not torch.equal(hrs[i], hrs[j]) for i in range(len(hrs)) for j in range(i + 1, len(hrs))
    )
    assert differ, "A.RandomCrop must yield varying HR crops across calls"


# ---------------------------------------------------------------------------
# TrainDataset LMDB HR cache (INIT.13)
# ---------------------------------------------------------------------------


def test_train_dataset_getitem_matches_source_pixels(varied_size_rgb_image_dir: Path):
    """The cached-and-cropped HR tensor must reproduce the source image's own
    pixels for the cropped region -- not merely the right shape. Guards
    against an off-by-header-offset or transposed (H, W) bug in the zero-copy
    read path, which shape/range assertions alone would miss."""
    ds = TrainDataset(
        img_dir=varied_size_rgb_image_dir,
        scale=2,
        hr_crop_size=16,
        cache_dir=varied_size_rgb_image_dir / ".lmdb_cache",
        build_num_workers=1,
    )
    lr, hr = ds[1]  # 48x32 image
    assert hr.shape == (3, 16, 16)
    assert lr.shape == (3, 8, 8)

    hr_uint8 = (hr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    source = np.array(Image.open(ds.img_paths[1]).convert("RGB"))
    # The crop must appear verbatim somewhere in the source image (a random
    # offset, but an exact contiguous 16x16 block of it).
    found = any(
        np.array_equal(source[top : top + 16, left : left + 16, :], hr_uint8)
        for top in range(0, 48 - 16 + 1)
        for left in range(0, 32 - 16 + 1)
    )
    assert found, "cropped HR tensor does not match any 16x16 block of its source image"


def test_train_dataset_varied_image_sizes_round_trip(varied_size_rgb_image_dir: Path):
    """Each image's own (H, W) header must be used to reshape its bytes --
    a fixed-constant reshape (as SRCNN uses, safely, for same-size sub-images)
    would corrupt every non-square image here."""
    ds = TrainDataset(
        img_dir=varied_size_rgb_image_dir,
        scale=2,
        hr_crop_size=16,
        cache_dir=varied_size_rgb_image_dir / ".lmdb_cache",
        build_num_workers=1,
    )
    for i in range(3):
        lr, hr = ds[i]
        assert hr.shape == (3, 16, 16)
        assert lr.shape == (3, 8, 8)


def test_train_dataset_cache_reuse_skips_rebuild(tiny_rgb_image_dir: Path):
    """Second instantiation with the same file set must not re-decode images."""
    cache_dir = tiny_rgb_image_dir / ".lmdb_cache"
    TrainDataset(
        img_dir=tiny_rgb_image_dir,
        scale=2,
        hr_crop_size=16,
        cache_dir=cache_dir,
        build_num_workers=1,
    )
    with patch("sisr.datasets.srresnet._process_hr_image") as mock_proc:
        TrainDataset(
            img_dir=tiny_rgb_image_dir,
            scale=2,
            hr_crop_size=16,
            cache_dir=cache_dir,
            build_num_workers=1,
        )
        mock_proc.assert_not_called()


def test_train_dataset_cache_independent_of_crop_params(tiny_rgb_image_dir: Path):
    """The cache stores whole images, so changing hr_crop_size/crops_per_image/
    scale over the same file set must NOT trigger a rebuild (contrast SRCNN,
    whose checksum bakes in sub_img_size/stride/scale/blur_sigma)."""
    cache_dir = tiny_rgb_image_dir / ".lmdb_cache"
    TrainDataset(
        img_dir=tiny_rgb_image_dir,
        scale=2,
        hr_crop_size=16,
        crops_per_image=1,
        cache_dir=cache_dir,
        build_num_workers=1,
    )
    with patch("sisr.datasets.srresnet._process_hr_image") as mock_proc:
        TrainDataset(
            img_dir=tiny_rgb_image_dir,
            scale=4,
            hr_crop_size=32,
            crops_per_image=5,
            cache_dir=cache_dir,
            build_num_workers=1,
        )
        mock_proc.assert_not_called()


def test_train_dataset_crops_per_image_reuses_one_cached_read(tiny_rgb_image_dir: Path):
    """crops_per_image=N must decode each image exactly once at build time --
    not once per crop -- regardless of how many items are drawn."""
    import sisr.datasets.srresnet as srresnet_mod

    with patch(
        "sisr.datasets.srresnet._process_hr_image", wraps=srresnet_mod._process_hr_image
    ) as mock_proc:
        ds = TrainDataset(
            img_dir=tiny_rgb_image_dir,
            scale=2,
            hr_crop_size=16,
            crops_per_image=5,
            cache_dir=tiny_rgb_image_dir / ".lmdb_cache",
            build_num_workers=1,
        )
        assert mock_proc.call_count == 3  # one decode per image, not per crop
    assert len(ds) == 3 * 5
    for i in range(len(ds)):
        lr, hr = ds[i]
        assert hr.shape == (3, 16, 16)


def test_train_dataset_missing_key_raises_keyerror(tiny_rgb_image_dir: Path):
    """A missing LMDB key (corrupt/incomplete cache) must surface as a
    KeyError naming the key, not a cryptic struct.error from unpacking None.

    __getitem__ maps idx via modulo into always-valid image indices, so an
    out-of-range idx (SRCNN's approach) can't reach this path here -- instead
    force get_buffer to report a miss, as a genuinely stale cache would."""
    ds = TrainDataset(
        img_dir=tiny_rgb_image_dir,
        scale=2,
        hr_crop_size=16,
        cache_dir=tiny_rgb_image_dir / ".lmdb_cache",
        build_num_workers=1,
    )

    @contextmanager
    def _always_missing(key):
        yield None

    with patch.object(ds._cache, "get_buffer", side_effect=_always_missing):
        with pytest.raises(KeyError, match=r"hr_\d{8}"):
            ds[0]


def test_train_dataset_build_num_workers_1_builds_inline(tiny_rgb_image_dir: Path):
    """build_num_workers=1 must thread through to an inline LMDB build (no
    ProcessPoolExecutor), safe inside a test/xdist worker."""
    with patch("sisr.cache.ProcessPoolExecutor") as mock_pool:
        ds = TrainDataset(
            img_dir=tiny_rgb_image_dir,
            scale=2,
            hr_crop_size=16,
            cache_dir=tiny_rgb_image_dir / ".lmdb_cache",
            build_num_workers=1,
        )
    mock_pool.assert_not_called()
    lr, hr = ds[0]
    assert hr.shape == (3, 16, 16)


def test_train_dataset_checksum_ignores_crop_params(tiny_rgb_image_dir: Path):
    """_compute_checksum must depend only on the file manifest, not on
    hr_crop_size/crops_per_image/scale."""
    ds_a = TrainDataset(
        img_dir=tiny_rgb_image_dir,
        scale=2,
        hr_crop_size=16,
        cache_dir=tiny_rgb_image_dir / ".lmdb_cache",
        build_num_workers=1,
    )
    ds_b = TrainDataset(
        img_dir=tiny_rgb_image_dir,
        scale=4,
        hr_crop_size=32,
        crops_per_image=3,
        cache_dir=tiny_rgb_image_dir / ".lmdb_cache",
        build_num_workers=1,
    )
    assert ds_a._compute_checksum() == ds_b._compute_checksum()


# ---------------------------------------------------------------------------
# ValidationDataset (full image)
# ---------------------------------------------------------------------------


def test_validation_dataset_hr_cropped_to_multiple_of_scale(tiny_rgb_image_dir: Path):
    """36x36 image, scale=4 -> hr stays 36 (divisible), lr is 9x9, and the
    model's x4 of lr would land exactly back on hr."""
    ds = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=4)
    assert len(ds) == 3
    lr, hr = ds[0]
    assert hr.shape == (3, 36, 36)
    assert lr.shape == (3, 9, 9)
    assert lr.shape[-1] * 4 == hr.shape[-1]


def test_validation_dataset_no_images_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No images"):
        ValidationDataset(img_dir=tmp_path, scale=2)


def test_validation_dataset_is_deterministic(tiny_rgb_image_dir: Path):
    """Calling __getitem__(idx) twice must return identical tensors —
    the validation pipeline has no random elements."""
    ds = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=4)
    lr_a, hr_a = ds[0]
    lr_b, hr_b = ds[0]
    assert torch.equal(lr_a, lr_b), "validation LR must be deterministic"
    assert torch.equal(hr_a, hr_b), "validation HR must be deterministic"


# ---------------------------------------------------------------------------
# SRDataset contract (P3.6 / P5.4)
# ---------------------------------------------------------------------------


def test_srresnet_datasets_are_srdataset_subclasses():
    from sisr.datasets.base import SRDataset

    assert issubclass(TrainDataset, SRDataset)
    assert issubclass(ValidationDataset, SRDataset)


def test_srresnet_validation_skips_non_image_files(tiny_rgb_image_dir: Path):
    (tiny_rgb_image_dir / "notes.txt").write_text("not an image")
    (tiny_rgb_image_dir / "MANIFEST").write_text("extensionless")
    ds = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=2)
    assert len(ds) == 3
    assert all(p.suffix == ".png" for p in ds.img_paths)
