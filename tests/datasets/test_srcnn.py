from contextlib import contextmanager
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


def test_train_dataset_out_of_range_index_raises_indexerror(shared_srcnn_train_ds: TrainDataset):
    with pytest.raises(IndexError):
        shared_srcnn_train_ds[len(shared_srcnn_train_ds) + 100]
    with pytest.raises(IndexError):
        shared_srcnn_train_ds[-1]


def test_train_dataset_missing_key_raises_keyerror(tiny_rgb_image_dir: Path):
    """A missing LMDB key (a corrupt/incomplete cache) surfaces as a KeyError
    naming the key, not a cryptic numpy error from np.frombuffer(None).

    __getitem__ maps idx to an always-valid image index, so an out-of-range
    idx can't reach this path (see test above) -- instead force get_buffer to
    report a miss, as a genuinely stale cache would."""
    ds = _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8)

    @contextmanager
    def _always_missing(key):
        yield None

    with patch.object(ds._cache, "get_buffer", side_effect=_always_missing):
        with pytest.raises(KeyError, match=r"hr_\d{8}"):
            ds[0]


def test_train_dataset_cache_reuse_skips_rebuild(tiny_rgb_image_dir: Path):
    """Second instantiation with the same file set must not re-decode images."""
    _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8)
    with patch("sisr.datasets.srcnn._process_hr_image") as mock_proc:
        _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8)
        mock_proc.assert_not_called()


def test_train_dataset_cache_independent_of_grid_params(tiny_rgb_image_dir: Path):
    """The cache stores whole images, so changing subimg_size/stride/scale
    over the same file set must NOT trigger a rebuild (contrast the
    pre-HR-only cache, whose checksum baked all of those in)."""
    cache_dir = tiny_rgb_image_dir / ".lmdb_cache_shared_grid"
    _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8, scale=2, cache_dir=cache_dir)
    with patch("sisr.datasets.srcnn._process_hr_image") as mock_proc:
        ds_b = _make_train(
            tiny_rgb_image_dir,
            subimg_size=24,
            stride=10,
            scale=4,
            cache_dir=cache_dir,
        )
        mock_proc.assert_not_called()
    assert len(ds_b) > 0


def test_train_dataset_checksum_change_triggers_rebuild(tmp_path: Path):
    """Adding/removing a file changes the manifest, hence the checksum."""
    import numpy as np
    from PIL import Image as PILImage

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rng = np.random.default_rng(seed=2)
    arr = rng.integers(0, 256, size=(36, 36, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(img_dir / "img_00.png")
    ds_a = _make_train(img_dir, subimg_size=20, stride=8)
    checksum_a = ds_a._compute_checksum()

    arr2 = rng.integers(0, 256, size=(36, 36, 3), dtype=np.uint8)
    PILImage.fromarray(arr2).save(img_dir / "img_01.png")
    ds_b = _make_train(img_dir, subimg_size=20, stride=8, cache_dir=img_dir / ".lmdb_cache_train_2")
    assert ds_b._compute_checksum() != checksum_a


def test_train_dataset_checksum_ignores_grid_and_lr_params(shared_srcnn_train_ds: TrainDataset):
    """_compute_checksum must depend only on the file manifest (+ a format
    tag) -- not on subimg_size/stride/scale, since the HR-only cache stores
    raw pixels unaffected by any of them."""
    import hashlib

    ds = shared_srcnn_train_ds
    file_manifest = ",".join(f"{p.name}:{p.stat().st_size}" for p in ds.img_paths)
    expected_canonical = "|".join([file_manifest, "format=hr_rgb_v1"])
    expected = hashlib.sha256(expected_canonical.encode("utf-8")).hexdigest()
    assert ds._compute_checksum() == expected


def test_train_dataset_grid_params_do_not_affect_checksum(tiny_rgb_image_dir: Path):
    """The HR cache stores raw pixels only; degradation/grid parameters are
    applied at read time, so two datasets differing only in
    subimg_size/stride/scale must produce the same checksum."""
    ds_a = _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8, scale=2)
    ds_b = _make_train(tiny_rgb_image_dir, subimg_size=24, stride=10, scale=4)
    assert ds_a._compute_checksum() == ds_b._compute_checksum()


def test_train_dataset_no_images_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No images"):
        TrainDataset(
            img_dir=tmp_path,
            subimg_size=20,
            stride=8,
            scale=2,
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
    with patch("sisr.utils.cache.ProcessPoolExecutor") as mock_pool:
        ds = _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8, build_num_workers=1)
    mock_pool.assert_not_called()
    assert len(ds) > 0
    lr, hr = ds[0]
    assert lr.shape == (3, 20, 20)
    assert hr.shape == (3, 20, 20)


def test_patch_grid_enumeration_matches_grid_dims(tiny_rgb_image_dir: Path):
    """_iter_patch_origins and _compute_grid must derive the sliding-window
    grid from the same _grid_dims helper, so their patch counts and ordering
    can never silently disagree."""
    from sisr.datasets.srcnn import _grid_dims, _iter_patch_origins

    ds = _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8, build_num_workers=1)

    n_rows, n_cols = _grid_dims(36, 36, 2, 20, 8)
    assert (n_rows, n_cols) == (3, 3)  # (36-20)//8 + 1 = 3 positions per axis

    origins = list(_iter_patch_origins(36, 36, 2, 20, 8))
    assert len(origins) == n_rows * n_cols == 9
    assert origins[0] == (0, 0)
    assert (16, 16) in origins

    assert ds._total_patches == len(ds.img_paths) * 9


def test_grid_index_mapping_matches_iteration_order(tiny_rgb_image_dir: Path):
    """TrainDataset.__getitem__'s O(1) index -> (top, left) lookup (bisect +
    divmod over _img_offsets/_img_n_cols) must agree, position-for-position,
    with _iter_patch_origins' explicit enumeration across every image in a
    multi-image dataset -- so an index can never silently point at the wrong
    sub-image (the single-source-of-truth guarantee this design established)."""
    import bisect

    from sisr.datasets.srcnn import _iter_patch_origins

    ds = _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8, scale=2)

    expected = []
    for path in ds.img_paths:
        img = Image.open(path)
        w, h = img.size
        img.close()
        expected.extend(_iter_patch_origins(h, w, ds.scale, ds.sub_img_size, ds.stride))
    assert len(expected) == len(ds) == ds._total_patches

    for idx, (exp_top, exp_left) in enumerate(expected):
        img_idx = bisect.bisect_right(ds._img_offsets, idx) - 1
        local_idx = idx - ds._img_offsets[img_idx]
        row, col = divmod(local_idx, ds._img_n_cols[img_idx])
        assert (row * ds.stride, col * ds.stride) == (exp_top, exp_left)


def test_grid_index_mapping_across_differently_sized_images(tmp_path: Path):
    """Same guarantee, but with images of *different* sizes and uneven grids.

    The uniform-fixture test above cannot catch a bug in the bisect/_img_n_cols
    interaction, because every image contributes an identical patch count and
    column stride. Deliberately prime-ish, non-square dimensions that do not
    divide evenly by the stride make each image's grid a different shape.
    """
    import bisect

    from sisr.datasets.srcnn import _iter_patch_origins

    rng = np.random.default_rng(7)
    for i, (h, w) in enumerate([(37, 53), (64, 41), (29, 29), (100, 67)]):
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"img_{i:02d}.png")

    ds = _make_train(tmp_path, subimg_size=20, stride=7, scale=3)

    expected = []
    for path in ds.img_paths:
        img = Image.open(path)
        w, h = img.size
        img.close()
        expected.extend(_iter_patch_origins(h, w, ds.scale, ds.sub_img_size, ds.stride))
    assert len(expected) == len(ds) == ds._total_patches
    assert len(set(ds._img_n_cols)) > 1, "fixture failed to produce differing grid widths"

    for idx, (exp_top, exp_left) in enumerate(expected):
        img_idx = bisect.bisect_right(ds._img_offsets, idx) - 1
        local_idx = idx - ds._img_offsets[img_idx]
        row, col = divmod(local_idx, ds._img_n_cols[img_idx])
        assert (row * ds.stride, col * ds.stride) == (exp_top, exp_left)


def test_train_dataset_hr_subimage_matches_exact_grid_position(tiny_rgb_image_dir: Path):
    """The cached-and-sliced HR sub-image must equal the source image's own
    pixels at its deterministic (top, left) grid position -- not merely
    'somewhere' in the image, since (unlike SRResNet) SRCNN's positions are
    fixed, not random."""
    from sisr.datasets.srcnn import _iter_patch_origins

    ds = _make_train(tiny_rgb_image_dir, subimg_size=20, stride=8, scale=2)
    origins = list(_iter_patch_origins(36, 36, 2, 20, 8))
    source = np.array(Image.open(ds.img_paths[0]).convert("RGB"))

    for idx, (top, left) in enumerate(origins):  # first image's patches only
        _, hr = ds[idx]
        hr_uint8 = (hr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        expected = source[top : top + 20, left : left + 20, :]
        assert np.array_equal(hr_uint8, expected), f"patch {idx} mismatch at ({top}, {left})"


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


def test_validation_dataset_is_deterministic(tiny_rgb_image_dir: Path):
    """Calling __getitem__(idx) twice must return identical tensors —
    the validation pipeline has no random elements."""
    ds = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=2)
    lr_a, hr_a = ds[0]
    lr_b, hr_b = ds[0]
    assert torch.equal(lr_a, lr_b), "validation LR must be deterministic"
    assert torch.equal(hr_a, hr_b), "validation HR must be deterministic"


# ---------------------------------------------------------------------------
# SRDataset contract
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


# ---------------------------------------------------------------------------
# modcrop: the authors' degradation order
# ---------------------------------------------------------------------------


@pytest.fixture
def non_divisible_image_dir(tmp_path: Path) -> Path:
    """One 481x321 image -- BSD100's exact size, where 481 % 3 == 1.

    The shared fixture is 36x36, divisible by 2/3/4, so it cannot see this
    defect at all. At x3 most standard benchmark images are NOT divisible,
    so the affected case is the common one rather than the corner.
    """
    rng = np.random.default_rng(seed=7)
    arr = rng.integers(0, 256, size=(321, 481, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "bsd_like.png")
    return tmp_path


def test_validation_hr_is_cropped_to_a_multiple_of_scale(non_divisible_image_dir: Path):
    """The released demo_SR.m applies `modcrop` to the ground truth BEFORE the
    bicubic round trip:

        im_gnd = modcrop(im, up_scale);
        im_l   = imresize(im_gnd, 1/up_scale, 'bicubic');
        im_b   = imresize(im_l,   up_scale,   'bicubic');

    Without it a 481-wide image is downsampled to 160 columns and stretched
    back over 481, so the LR grid does not align with the HR one."""
    ds = ValidationDataset(img_dir=non_divisible_image_dir, scale=3)
    lr, hr = ds[0]
    assert hr.shape[-2:] == (321, 480), "HR must be modcropped to a multiple of scale"
    assert lr.shape == hr.shape, "SRCNN is pre-upsampled: LR and HR share a size"


def test_validation_lr_matches_the_authors_degradation_order(non_divisible_image_dir: Path):
    """Byte-exact against modcrop-then-degrade, computed independently here."""
    from sisr.datasets.srcnn import _degrade

    arr = np.array(Image.open(next(non_divisible_image_dir.iterdir())).convert("RGB"))
    h, w = arr.shape[:2]
    expected_hr = arr[: h - h % 3, : w - w % 3]
    expected_lr = _degrade(expected_hr, 3)

    lr, hr = ValidationDataset(img_dir=non_divisible_image_dir, scale=3)[0]
    got_hr = (hr.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    got_lr = (lr.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    assert np.array_equal(got_hr, expected_hr)
    assert np.array_equal(got_lr, expected_lr)


def test_train_and_validation_share_one_modcrop_implementation():
    """The convention had three sites and two answers. It must now have one
    owner that the training grid and the validation loader both call."""
    from sisr.datasets.srcnn import _modcrop, _modcrop_extent

    assert _modcrop_extent(481, 3) == 480
    assert _modcrop_extent(321, 3) == 321  # already a multiple -- unchanged
    assert _modcrop(np.zeros((321, 481, 3), np.uint8), 3).shape == (321, 480, 3)


def test_validation_untouched_when_already_divisible(tiny_rgb_image_dir: Path):
    """36x36 divides by 2, 3 and 4, so no shipped-size behaviour changes."""
    for scale in (2, 3, 4):
        lr, hr = ValidationDataset(img_dir=tiny_rgb_image_dir, scale=scale)[0]
        assert hr.shape[-2:] == (36, 36)
        assert lr.shape == hr.shape
