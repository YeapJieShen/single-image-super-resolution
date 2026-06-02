from pathlib import Path

import pytest
import torch

from sisr.datasets.srresnet import TrainDataset, ValidationDataset


# ---------------------------------------------------------------------------
# TrainDataset (random-crop)
# ---------------------------------------------------------------------------

def test_train_dataset_getitem_lr_is_downscaled_hr(tiny_rgb_image_dir: Path):
    ds = TrainDataset(img_dir=tiny_rgb_image_dir, scale=4, hr_crop_size=16)
    lr, hr = ds[0]
    assert hr.shape == (3, 16, 16)
    assert lr.shape == (3, 4, 4)  # 16 // 4
    assert lr.dtype == torch.float32 and hr.dtype == torch.float32
    assert 0.0 <= lr.min() <= lr.max() <= 1.0
    assert 0.0 <= hr.min() <= hr.max() <= 1.0


def test_train_dataset_len_scales_with_crops_per_image(tiny_rgb_image_dir: Path):
    ds = TrainDataset(
        img_dir=tiny_rgb_image_dir, scale=2, hr_crop_size=16, crops_per_image=4
    )
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
        not torch.equal(hrs[i], hrs[j])
        for i in range(len(hrs)) for j in range(i + 1, len(hrs))
    )
    assert differ, "A.RandomCrop must yield varying HR crops across calls"


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
