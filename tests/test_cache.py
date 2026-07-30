import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import lmdb
import pytest

from sisr.cache import LMDBCache, LMDBCacheBuildContext

_MAP_SIZE = 16 * 1024 * 1024  # 16 MiB — plenty for tiny test caches


def _make_build_fn(n: int, value_prefix: bytes = b"v"):
    """Returns a build_fn that writes n keys named key_0..key_{n-1}."""

    def build(ctx: LMDBCacheBuildContext) -> None:
        ctx.write_batch([(f"key_{i}", value_prefix + str(i).encode()) for i in range(n)])

    return build


# ---------------------------------------------------------------------------
# Import weight
# ---------------------------------------------------------------------------


def test_cache_module_imports_no_torch():
    """sisr.cache must not pull torch in, directly or transitively.

    parallel_build fans out over a ProcessPoolExecutor, and on spawn platforms
    each worker re-imports the module tree holding its process_fn. A torch
    import reachable from here costs every worker several seconds for a
    dependency the build path never calls, so guard it: a fresh interpreter is
    required because the test session itself has torch loaded already.
    """
    probe = "import sys, sisr.cache; print('torch' in sys.modules)"
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "False", (
        "sisr.cache now imports torch (directly or transitively). Every spawned "
        "LMDB build worker pays that import for nothing — move the torch-dependent "
        "code to sisr.colorspace or another module."
    )


# ---------------------------------------------------------------------------
# LMDBCache
# ---------------------------------------------------------------------------


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
    with patch("sisr.cache.lmdb.open", side_effect=lmdb.Error("corrupt")):
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
    with patch("sisr.cache.lmdb.open", side_effect=ValueError("unexpected")):
        with pytest.raises(ValueError, match="unexpected"):
            cache._try_load("abc")


# ---------------------------------------------------------------------------
# LMDBCacheBuildContext.parallel_build
# ---------------------------------------------------------------------------


def _sq_process_fn(item: int) -> list[tuple[str, bytes]]:
    """Top-level (picklable) worker: squares *item* into one keyed pair.

    Must be module-level so ProcessPoolExecutor can pickle it on spawn platforms.
    """
    return [(f"n_{item}", str(item * item).encode())]


def test_parallel_build_persists_every_submitted_item(tmp_path: Path):
    """parallel_build must submit every item to the pool and persist each
    worker's returned pairs — previously only write_batch was covered, so the
    submit/collect loop was untested."""

    def build(ctx: LMDBCacheBuildContext) -> None:
        ctx.parallel_build(items=[2, 3, 4, 5], process_fn=_sq_process_fn, num_workers=2)

    cache = LMDBCache(
        cache_dir=tmp_path,
        name="pb",
        checksum="c",
        length=4,
        map_size=_MAP_SIZE,
        build_fn=build,
    )
    assert cache.length == 4
    assert cache.get("n_2") == b"4"
    assert cache.get("n_5") == b"25"


def test_parallel_build_single_worker_skips_process_pool(tmp_path: Path):
    """num_workers=1 must run the build inline (no ProcessPoolExecutor), so it
    is safe to nest inside a test/xdist worker without oversubscribing cores."""

    def build(ctx: LMDBCacheBuildContext) -> None:
        ctx.parallel_build(items=[2, 3, 4], process_fn=_sq_process_fn, num_workers=1)

    with patch("sisr.cache.ProcessPoolExecutor") as mock_pool:
        cache = LMDBCache(
            cache_dir=tmp_path,
            name="pb1",
            checksum="c",
            length=3,
            map_size=_MAP_SIZE,
            build_fn=build,
        )
    mock_pool.assert_not_called()
    assert cache.get("n_2") == b"4"
    assert cache.get("n_4") == b"16"


def test_parallel_build_none_workers_builds_single_item_inline(tmp_path: Path):
    """num_workers=None must resolve to min(os.cpu_count() or 1, n_items); a lone
    item yields <= 1 effective worker and therefore an inline build."""

    def build(ctx: LMDBCacheBuildContext) -> None:
        ctx.parallel_build(items=[7], process_fn=_sq_process_fn, num_workers=None)

    with patch("sisr.cache.ProcessPoolExecutor") as mock_pool:
        cache = LMDBCache(
            cache_dir=tmp_path,
            name="pbn",
            checksum="c",
            length=1,
            map_size=_MAP_SIZE,
            build_fn=build,
        )
    mock_pool.assert_not_called()
    assert cache.get("n_7") == b"49"
