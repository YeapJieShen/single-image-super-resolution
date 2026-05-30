from pathlib import Path
from unittest.mock import patch

import lmdb
import pytest
import torch

from sisr.utils import (
    LMDBCache,
    LMDBCacheBuildContext,
    extract_model_input,
    reconstruct_sr_rgb,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


# ---------------------------------------------------------------------------
# Colorspace utilities
# ---------------------------------------------------------------------------

def test_rgb_to_ycbcr_shape_and_dtype():
    x = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    y = rgb_to_ycbcr(x)
    assert y.shape == (2, 3, 8, 8)
    assert y.dtype == torch.float32


def test_ycbcr_to_rgb_shape_and_dtype():
    x = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    y = ycbcr_to_rgb(x)
    assert y.shape == (2, 3, 8, 8)
    assert y.dtype == torch.float32


def test_rgb_to_ycbcr_known_values_white():
    # White: (1, 1, 1) -> Y=1, Cb=0.5, Cr=0.5
    x = torch.ones(1, 3, 1, 1)
    y = rgb_to_ycbcr(x)
    expected = torch.tensor([[[[1.0]], [[0.5]], [[0.5]]]])
    assert torch.allclose(y, expected, atol=1e-5)


def test_rgb_to_ycbcr_known_values_black():
    # Black: (0, 0, 0) -> Y=0, Cb=0.5, Cr=0.5
    x = torch.zeros(1, 3, 1, 1)
    y = rgb_to_ycbcr(x)
    expected = torch.tensor([[[[0.0]], [[0.5]], [[0.5]]]])
    assert torch.allclose(y, expected, atol=1e-5)


def test_rgb_to_ycbcr_known_values_red():
    # Pure red: (1, 0, 0) -> Y=0.299, Cb=0.5−0.169, Cr=0.5+0.500
    x = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
    y = rgb_to_ycbcr(x)
    expected = torch.tensor([[[[0.299]], [[0.331]], [[1.000]]]])
    assert torch.allclose(y, expected, atol=1e-5)


def test_round_trip_within_coefficient_precision():
    # BT.601 coefficients are 3-decimal-place; round-trip error floor ~5e-4.
    torch.manual_seed(0)
    x = torch.rand(1, 3, 16, 16)
    y = ycbcr_to_rgb(rgb_to_ycbcr(x))
    err = (y - x).abs().max().item()
    assert err < 5e-4


def test_extract_model_input_rgb_passthrough():
    x = torch.rand(1, 3, 4, 4)
    inp, aux = extract_model_input(x, "RGB")
    assert torch.equal(inp, x)
    assert aux is None


def test_extract_model_input_y_returns_y_and_full_ycbcr():
    x = torch.rand(1, 3, 4, 4)
    inp, aux = extract_model_input(x, "Y")
    assert inp.shape == (1, 1, 4, 4)
    assert aux is not None
    assert aux.shape == (1, 3, 4, 4)


def test_extract_model_input_ycbcr():
    x = torch.rand(1, 3, 4, 4)
    inp, aux = extract_model_input(x, "YCbCr")
    assert inp.shape == (1, 3, 4, 4)
    assert aux is None


def test_extract_model_input_unknown_raises():
    x = torch.rand(1, 3, 4, 4)
    with pytest.raises(ValueError):
        extract_model_input(x, "XYZ")


def test_reconstruct_sr_rgb_rgb_passthrough():
    sr = torch.rand(1, 3, 4, 4)
    out = reconstruct_sr_rgb(sr, lr_ycbcr=None, model_colorspace="RGB")
    assert torch.equal(out, sr)


def test_reconstruct_sr_rgb_y_requires_lr_ycbcr():
    sr_y = torch.rand(1, 1, 4, 4)
    with pytest.raises(ValueError):
        reconstruct_sr_rgb(sr_y, lr_ycbcr=None, model_colorspace="Y")


def test_reconstruct_sr_rgb_y_centre_crops_chroma():
    """When SR is spatially smaller than LR (valid padding), reconstruct must
    centre-crop the LR's Cb/Cr to match SR's spatial size."""
    lr_ycbcr = torch.rand(1, 3, 8, 8)
    sr_y = torch.zeros(1, 1, 4, 4)
    out = reconstruct_sr_rgb(sr_y, lr_ycbcr=lr_ycbcr, model_colorspace="Y")
    assert out.shape == (1, 3, 4, 4)


def test_reconstruct_sr_rgb_ycbcr_converts_to_rgb():
    sr = torch.rand(1, 3, 4, 4)
    out = reconstruct_sr_rgb(sr, lr_ycbcr=None, model_colorspace="YCbCr")
    assert out.shape == (1, 3, 4, 4)
    # ycbcr_to_rgb clamps to [0, 1].
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_reconstruct_sr_rgb_unknown_raises():
    sr = torch.rand(1, 3, 4, 4)
    with pytest.raises(ValueError):
        reconstruct_sr_rgb(sr, lr_ycbcr=None, model_colorspace="XYZ")


# ---------------------------------------------------------------------------
# LMDBCache
# ---------------------------------------------------------------------------

_MAP_SIZE = 16 * 1024 * 1024  # 16 MiB — plenty for tiny test caches


def _make_build_fn(n: int, value_prefix: bytes = b"v"):
    """Returns a build_fn that writes n keys named key_0..key_{n-1}."""
    def build(ctx: LMDBCacheBuildContext) -> None:
        ctx.write_batch([(f"key_{i}", value_prefix + str(i).encode()) for i in range(n)])
    return build


def test_lmdb_build_and_read(tmp_path: Path):
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=5,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(5),
    )
    assert cache.length == 5
    env = cache.get_env()
    with env.begin() as txn:
        assert bytes(txn.get(b"key_0")) == b"v0"
        assert bytes(txn.get(b"key_4")) == b"v4"


def test_lmdb_get_method(tmp_path: Path):
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=3,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(3),
    )
    assert cache.get("key_1") == b"v1"
    assert cache.get("missing") is None


def test_lmdb_get_batch(tmp_path: Path):
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=3,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(3),
    )
    out = cache.get_batch(["key_0", "missing", "key_2"])
    assert out == [b"v0", None, b"v2"]


def test_lmdb_checksum_skips_rebuild_when_unchanged(tmp_path: Path):
    """Re-instantiating with the same checksum must not call build_fn again."""
    call_count = [0]

    def counting_build(ctx: LMDBCacheBuildContext) -> None:
        call_count[0] += 1
        ctx.write_batch([("key_0", b"v0")])

    LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="checksum_v1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=counting_build,
    )
    assert call_count[0] == 1
    LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="checksum_v1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=counting_build,
    )
    assert call_count[0] == 1, "build_fn must not run when checksum matches"


def test_lmdb_checksum_change_triggers_rebuild(tmp_path: Path):
    call_count = [0]

    def counting_build(ctx: LMDBCacheBuildContext) -> None:
        call_count[0] += 1
        ctx.write_batch([("key_0", b"v0")])

    LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="checksum_v1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=counting_build,
    )
    LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="checksum_v2",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=counting_build,
    )
    assert call_count[0] == 2, "checksum change must trigger rebuild"


def test_lmdb_metadata_round_trips(tmp_path: Path):
    """Metadata is stored as `__{key}__` -> str(value). bytes."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        metadata={"flavor": "vanilla", "scale": "3"},
        build_fn=_make_build_fn(1),
    )
    env = cache.get_env()
    with env.begin() as txn:
        assert bytes(txn.get(b"__flavor__")) == b"vanilla"
        assert bytes(txn.get(b"__scale__")) == b"3"
        assert bytes(txn.get(b"__checksum__")) == b"abc"
        assert bytes(txn.get(b"__length__")) == b"1"


def test_lmdb_missing_dir_with_no_build_fn_raises(tmp_path: Path):
    with pytest.raises(RuntimeError):
        LMDBCache(
            cache_dir=tmp_path,
            name="empty",
            checksum="abc",
            length=1,
            map_size=_MAP_SIZE,
            build_fn=None,
        )


def test_lmdb_try_load_swallows_lmdb_error_and_drops_cache(tmp_path: Path):
    """An unreadable cache (lmdb.Error) is treated as stale: _try_load returns
    False and removes the directory so it can be rebuilt."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    assert cache.path.exists()
    with patch("sisr.utils.lmdb.open", side_effect=lmdb.Error("corrupt")):
        assert cache._try_load("abc") is False
    assert not cache.path.exists()


def test_lmdb_try_load_propagates_unexpected_error(tmp_path: Path):
    """A non-LMDB/OS error must propagate rather than be misreported as a
    stale cache."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    with patch("sisr.utils.lmdb.open", side_effect=ValueError("unexpected")):
        with pytest.raises(ValueError, match="unexpected"):
            cache._try_load("abc")
