"""Multi-scale training data: a scale per patch (pre-upsampled), a scale per batch (native-LR)."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from sisr.datasets import derived_cache, native_lr, pre_upsampled
from sisr.datasets.sampling import ScaleBatchSampler

SCALES = [2, 3, 4]


@pytest.fixture
def mixed_size_image_dir(tmp_path: Path) -> Path:
    """Two images: one a multiple of lcm(2,3,4)=12 both ways, one that is not."""
    rng = np.random.default_rng(seed=3)
    for i, (h, w) in enumerate([(48, 48), (50, 44)]):
        Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)).save(
            tmp_path / f"img_{i}.png"
        )
    return tmp_path


def _u8(t: torch.Tensor) -> np.ndarray:
    return (t * 255).round().byte().permute(1, 2, 0).numpy()


def _locate(whole: np.ndarray, crop: np.ndarray) -> tuple[int, int] | None:
    size = crop.shape[0]
    for top in range(whole.shape[0] - size + 1):
        for left in range(whole.shape[1] - size + 1):
            if (whole[top : top + size, left : left + size] == crop).all():
                return top, left
    return None


# --- pre_upsampled: one scale per patch ------------------------------------------------


def _scale_of_patch(ds, image_arrays, lr, hr):
    """Which scale produced this (lr, hr) pair: the one whose whole-image degradation,
    sliced at the patch's own position, equals ``lr`` byte for byte."""
    hr_u8 = _u8(hr)
    for arr in image_arrays:
        pos = _locate(arr, hr_u8)
        if pos is None:
            continue
        top, left = pos
        size = hr_u8.shape[0]
        matches = [
            s
            for s in ds.scales
            if (
                derived_cache.derive(arr, "bicubic", s)[top : top + size, left : left + size]
                == _u8(lr)
            ).all()
        ]
        return matches
    raise AssertionError("patch not found in any source image")


def test_pre_upsampled_train_draws_every_scale_and_each_pair_is_exact(
    mixed_size_image_dir: Path,
):
    """Over many draws all three scales occur, and every (lr, hr) pair is byte-identical to
    the whole-image degradation at the scale it used -- on an image that is not a multiple of
    4 or 3, where a grid cut by one scale's modcrop would run off the others'."""
    ds = pre_upsampled.TrainDataset(mixed_size_image_dir, subimg_size=12, stride=4, scale=SCALES)
    arrays = [np.array(Image.open(p).convert("RGB")) for p in ds.img_paths]
    random.seed(0)

    seen = set()
    for _ in range(12):
        for idx in range(len(ds)):
            lr, hr = ds[idx]
            matches = _scale_of_patch(ds, arrays, lr, hr)
            assert matches, f"item {idx} matches no scale's whole-image degradation"
            seen.update(matches)

    assert seen == set(SCALES)


def test_pre_upsampled_single_scale_list_is_the_int(mixed_size_image_dir: Path):
    """``scale=[3]`` is ``scale=3``: same items, and the same RNG stream (no scale draw)."""
    d = mixed_size_image_dir
    a = pre_upsampled.TrainDataset(d, subimg_size=12, stride=12, scale=3, cache_dir=d / "a")
    b = pre_upsampled.TrainDataset(d, subimg_size=12, stride=12, scale=[3], cache_dir=d / "b")

    assert len(a) == len(b) and a.scale == b.scale == 3
    for idx in range(len(a)):
        for x, y in zip(a[idx], b[idx], strict=True):
            assert torch.equal(x, y)


def test_pre_upsampled_multi_scale_exposes_its_scales(mixed_size_image_dir: Path):
    ds = pre_upsampled.TrainDataset(mixed_size_image_dir, subimg_size=12, stride=12, scale=SCALES)

    assert ds.scales == (2, 3, 4) and ds.scale == [2, 3, 4]
    assert ds.scale_per_batch is False


@pytest.mark.parametrize("bad", [[], [2, 2], [0, 2], [2, True]])
def test_train_datasets_reject_malformed_scale_lists(mixed_size_image_dir: Path, bad):
    with pytest.raises(ValueError, match="scale"):
        pre_upsampled.TrainDataset(mixed_size_image_dir, subimg_size=12, stride=12, scale=bad)
    with pytest.raises(ValueError, match="scale"):
        native_lr.TrainDataset(mixed_size_image_dir, scale=bad, hr_crop_size=12)


# --- native_lr: one scale per batch ----------------------------------------------------


def test_native_lr_crop_must_divide_every_scale_and_the_error_names_the_offender(
    mixed_size_image_dir: Path,
):
    with pytest.raises(ValueError, match=r"\(18\).*scale \(4\)"):
        native_lr.TrainDataset(mixed_size_image_dir, scale=[2, 3, 4], hr_crop_size=18)


def test_native_lr_pair_shapes_follow_the_requested_scale(mixed_size_image_dir: Path):
    ds = native_lr.TrainDataset(mixed_size_image_dir, scale=SCALES, hr_crop_size=24)

    for s in SCALES:
        lr, hr = ds[(0, s)]
        assert hr.shape == (3, 24, 24) and lr.shape == (3, 24 // s, 24 // s)
    assert ds[0][0].shape[-1] == 24 // 2, "a bare index uses the first scale"
    with pytest.raises(ValueError, match="not one of"):
        ds[(0, 5)]


def test_native_lr_single_scale_list_is_the_int(mixed_size_image_dir: Path):
    d = mixed_size_image_dir
    a = native_lr.TrainDataset(d, scale=2, hr_crop_size=24, cache_dir=d / "a")
    b = native_lr.TrainDataset(d, scale=[2], hr_crop_size=24, cache_dir=d / "b")
    random.seed(1)
    ref = [a[i] for i in range(len(a))]
    random.seed(1)
    got = [b[i] for i in range(len(b))]

    assert b.scale == 2 and b.scale_per_batch is False
    for (x1, y1), (x2, y2) in zip(ref, got, strict=True):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def _loader(ds, batch_size=2, drop_last=False):
    sampler = ScaleBatchSampler(
        RandomSampler(ds), ds.scales, batch_size=batch_size, drop_last=drop_last
    )
    return DataLoader(ds, batch_sampler=sampler)


def test_every_batch_has_one_scale_and_all_scales_appear(mixed_size_image_dir: Path):
    ds = native_lr.TrainDataset(
        mixed_size_image_dir, scale=SCALES, hr_crop_size=24, crops_per_image=6
    )
    torch.manual_seed(0)
    ratios = []

    for _ in range(3):
        for lr, hr in _loader(ds, batch_size=3):
            per_sample = {hr.shape[-1] // lr.shape[-1]}
            assert len(per_sample) == 1  # collation would already have failed on mixed shapes
            ratios.append(per_sample.pop())

    assert set(ratios) == set(SCALES)


def test_scale_batch_sampler_groups_and_tags_one_scale_per_batch():
    sampler = ScaleBatchSampler(range(7), [2, 4], batch_size=3)  # type: ignore[arg-type]
    torch.manual_seed(0)

    batches = list(sampler)

    assert [len(b) for b in batches] == [3, 3, 1] and len(sampler) == 3
    assert [i for b in batches for i, _ in b] == list(range(7))
    for b in batches:
        assert len({s for _, s in b}) == 1 and b[0][1] in (2, 4)
    assert len(list(ScaleBatchSampler(range(7), [2], 3, drop_last=True))) == 2  # type: ignore[arg-type]


def test_scale_batch_sampler_is_reproducible_under_the_torch_seed():
    def run():
        torch.manual_seed(5)
        return list(ScaleBatchSampler(range(20), [2, 3, 4], batch_size=2))  # type: ignore[arg-type]

    assert run() == run()


def test_scale_batch_sampler_keeps_distributed_shards_disjoint(mixed_size_image_dir: Path):
    ds = native_lr.TrainDataset(
        mixed_size_image_dir, scale=SCALES, hr_crop_size=24, crops_per_image=2
    )
    per_rank = []
    for rank in range(2):
        index_sampler = DistributedSampler(ds, num_replicas=2, rank=rank, shuffle=True)
        batches = ScaleBatchSampler(index_sampler, ds.scales, batch_size=2)
        assert batches.sampler is index_sampler  # where Lightning calls set_epoch
        per_rank.append({i for b in batches for i, _ in b})

    assert per_rank[0].isdisjoint(per_rank[1])
    assert per_rank[0] | per_rank[1] == set(range(len(ds)))


@pytest.mark.parametrize("kwargs", [{"scales": []}, {"batch_size": 0}])
def test_scale_batch_sampler_rejects_nonsense(kwargs):
    args = {"sampler": range(3), "scales": [2], "batch_size": 2} | kwargs
    with pytest.raises(ValueError):
        ScaleBatchSampler(**args)  # type: ignore[arg-type]
