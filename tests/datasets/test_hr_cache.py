"""Tests for the architecture-neutral raw-HR cache module.

Covers the shared primitives directly (process_hr_image / compute_checksum /
estimate_map_size), the import-weight guarantee for the worker entry point's
own module, and cross-architecture cache sharing/pixel equivalence between
SRCNN's and SRResNet's train datasets.
"""

import hashlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from sisr.datasets.hr_cache import HEADER, compute_checksum, estimate_map_size, process_hr_image

# ---------------------------------------------------------------------------
# Import weight
# ---------------------------------------------------------------------------


def test_hr_cache_module_imports_no_torch():
    """sisr.datasets.hr_cache must not pull torch in, directly or transitively.

    This is the module process_hr_image actually lives in, and therefore the
    module a spawned ProcessPoolExecutor build worker re-imports to unpickle
    it -- regardless of which dataset module (srcnn/srresnet, both of which
    do import torch) merely re-exports a reference to it. Asserting this on
    sisr.cache alone would prove the wrong thing: the worker's entry point is
    this module, not that one.
    """
    probe = "import sys, sisr.datasets.hr_cache; print('torch' in sys.modules)"
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "False", (
        "sisr.datasets.hr_cache now imports torch (directly or transitively). "
        "Every spawned LMDB build worker pays that import for nothing."
    )


# ---------------------------------------------------------------------------
# process_hr_image / compute_checksum / estimate_map_size
# ---------------------------------------------------------------------------


def test_process_hr_image_returns_headered_value(tmp_path: Path):
    rng = np.random.default_rng(seed=0)
    arr = rng.integers(0, 256, size=(12, 20, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    Image.fromarray(arr).save(path)

    [(key, value)] = process_hr_image(path, 3)

    assert key == "hr_00000003"
    h, w = HEADER.unpack_from(value, 0)
    assert (h, w) == (12, 20)
    payload = np.frombuffer(value, dtype=np.uint8, offset=HEADER.size).reshape(h, w, 3)
    assert np.array_equal(payload, arr)


def test_compute_checksum_depends_on_manifest_and_format_tag(tmp_path: Path):
    rng = np.random.default_rng(seed=1)
    paths = []
    for i in range(2):
        arr = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)

    file_manifest = ",".join(f"{p.name}:{p.stat().st_size}" for p in paths)
    expected = hashlib.sha256(f"{file_manifest}|format=hr_rgb_v1".encode()).hexdigest()
    assert compute_checksum(paths) == expected


def test_estimate_map_size_includes_header_and_slack():
    sizes = [(10, 20), (5, 5)]
    expected_total = (HEADER.size + 10 * 20 * 3) + (HEADER.size + 5 * 5 * 3)
    assert estimate_map_size(sizes) == max(int(expected_total * 1.1), 64 * 1024 * 1024)


def test_estimate_map_size_floors_at_64_mib():
    assert estimate_map_size([(1, 1)]) == 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# Cross-architecture sharing (Task A)
# ---------------------------------------------------------------------------


@pytest.fixture
def varied_size_rgb_image_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(seed=2)
    for i, (h, w) in enumerate([(40, 40), (48, 32), (30, 50)]):
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"img_{i:02d}.png")
    return tmp_path


def test_srresnet_build_is_reused_by_srcnn_same_files(varied_size_rgb_image_dir: Path):
    """Building via one architecture must be reused, not rebuilt, by the other
    over the identical file set -- the point of unifying the cache."""
    from sisr.datasets.srcnn import TrainDataset as SRCNNTrainDataset
    from sisr.datasets.srresnet import TrainDataset as SRResNetTrainDataset

    cache_dir = varied_size_rgb_image_dir / ".lmdb_cache"

    srresnet_ds = SRResNetTrainDataset(
        img_dir=varied_size_rgb_image_dir,
        scale=2,
        hr_crop_size=16,
        cache_dir=cache_dir,
        build_num_workers=1,
    )

    with patch("sisr.datasets.srcnn._process_hr_image") as mock_proc:
        srcnn_ds = SRCNNTrainDataset(
            img_dir=varied_size_rgb_image_dir,
            subimg_size=16,
            stride=8,
            scale=2,
            cache_dir=cache_dir,
            build_num_workers=1,
        )
        mock_proc.assert_not_called()

    assert srcnn_ds._cache.path == srresnet_ds._cache.path
    assert srcnn_ds._compute_checksum() == srresnet_ds._compute_checksum()


def test_srcnn_build_is_reused_by_srresnet_same_files(varied_size_rgb_image_dir: Path):
    """Same guarantee in the other build order -- whichever architecture
    builds first, the other must find and reuse it."""
    from sisr.datasets.srcnn import TrainDataset as SRCNNTrainDataset
    from sisr.datasets.srresnet import TrainDataset as SRResNetTrainDataset

    cache_dir = varied_size_rgb_image_dir / ".lmdb_cache"

    srcnn_ds = SRCNNTrainDataset(
        img_dir=varied_size_rgb_image_dir,
        subimg_size=16,
        stride=8,
        scale=2,
        cache_dir=cache_dir,
        build_num_workers=1,
    )

    with patch("sisr.datasets.srresnet._process_hr_image") as mock_proc:
        srresnet_ds = SRResNetTrainDataset(
            img_dir=varied_size_rgb_image_dir,
            scale=2,
            hr_crop_size=16,
            cache_dir=cache_dir,
            build_num_workers=1,
        )
        mock_proc.assert_not_called()

    assert srcnn_ds._cache.path == srresnet_ds._cache.path


def test_shared_cache_pixel_equivalence_across_architectures(varied_size_rgb_image_dir: Path):
    """Sub-images/crops read through the shared cache by EITHER architecture
    must be byte-identical to the raw source pixels -- proving the single
    cache serves correct data down both read paths (header-skip offsets
    differ: SRCNN ignores the header content, SRResNet parses it).

    The two datasets are used sequentially, not concurrently: LMDB itself
    refuses to open the same environment path twice in one process (a
    same-process concern only -- separate training processes, the actual
    production scenario, each open their own handle just fine), so the first
    dataset's env is closed before the second is constructed.
    """
    from sisr.datasets.srcnn import TrainDataset as SRCNNTrainDataset
    from sisr.datasets.srresnet import TrainDataset as SRResNetTrainDataset

    cache_dir = varied_size_rgb_image_dir / ".lmdb_cache"
    source = np.array(Image.open(sorted(varied_size_rgb_image_dir.glob("*.png"))[0]).convert("RGB"))

    srcnn_ds = SRCNNTrainDataset(
        img_dir=varied_size_rgb_image_dir,
        subimg_size=16,
        stride=16,
        scale=2,
        cache_dir=cache_dir,
        build_num_workers=1,
    )
    # SRCNN's first sub-image is the deterministic (0, 0) corner.
    _, hr_srcnn = srcnn_ds[0]
    hr_srcnn_uint8 = (hr_srcnn.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    assert np.array_equal(hr_srcnn_uint8, source[:16, :16, :])
    srcnn_ds._cache.get_env().close()  # release before srresnet_ds opens the same path

    srresnet_ds = SRResNetTrainDataset(
        img_dir=varied_size_rgb_image_dir,
        scale=2,
        hr_crop_size=16,
        cache_dir=cache_dir,
        build_num_workers=1,
    )
    # SRResNet's crop is random, but must appear verbatim somewhere in the source.
    _, hr_srresnet = srresnet_ds[0]
    hr_srresnet_uint8 = (hr_srresnet.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    h, w = source.shape[:2]
    found = any(
        np.array_equal(source[top : top + 16, left : left + 16, :], hr_srresnet_uint8)
        for top in range(0, h - 16 + 1)
        for left in range(0, w - 16 + 1)
    )
    assert found, "SRResNet crop read through the shared cache does not match its source image"
