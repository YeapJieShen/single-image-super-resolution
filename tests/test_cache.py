import os
import subprocess
import sys
import time
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


def test_lmdb_get_buffer_yields_memoryview_matching_get(tmp_path: Path):
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=3,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(3),
    )
    with cache.get_buffer("key_1") as buf:
        assert isinstance(buf, memoryview)
        assert bytes(buf) == b"v1"


def test_lmdb_get_buffer_yields_none_for_missing_key(tmp_path: Path):
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    with cache.get_buffer("missing") as buf:
        assert buf is None


def test_lmdb_get_buffer_is_a_context_manager_not_a_plain_value(tmp_path: Path):
    """get_buffer() itself must return a context manager, not the buffer --
    misuse (retaining the view past its transaction) requires deliberately
    working around the API shape rather than just calling it normally."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    ctx = cache.get_buffer("key_0")
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")
    assert not isinstance(ctx, (bytes, memoryview))


def test_lmdb_get_buffer_does_not_leak_read_transactions(tmp_path: Path):
    """Each with-block must commit its transaction on exit rather than leaking
    it -- LMDB's default max_readers is 126, so many sequential get_buffer
    calls would raise lmdb.ReadersFullError if the context manager failed to
    close the transaction it opened."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    for _ in range(300):
        with cache.get_buffer("key_0") as buf:
            assert bytes(buf) == b"v0"


def test_lmdb_get_buffer_propagates_exception_and_stays_usable(tmp_path: Path):
    """An exception raised inside the with-block must propagate (the
    transaction aborts, it isn't swallowed), and the cache must remain
    readable afterward -- the aborted transaction must not corrupt or lock
    the read-only environment for subsequent reads."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    with pytest.raises(ValueError, match="boom"):
        with cache.get_buffer("key_0"):
            raise ValueError("boom")
    with cache.get_buffer("key_0") as buf:
        assert bytes(buf) == b"v0"


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


# ---------------------------------------------------------------------------
# Advisory build lock
# ---------------------------------------------------------------------------


def test_acquire_lock_uncontended_creates_sentinel_and_returns_true(tmp_path: Path):
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="lock",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    lock_path = cache._lock_path()
    assert not lock_path.exists(), "the build's own lock must be released once it completes"

    assert cache._acquire_lock("abc", poll_interval=0.01, timeout=0.05) is True
    assert lock_path.exists()
    assert lock_path.read_text() == str(os.getpid())


def test_release_lock_removes_sentinel_and_is_idempotent(tmp_path: Path):
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="lock",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    cache._acquire_lock("abc", poll_interval=0.01, timeout=0.05)
    assert cache._lock_path().exists()

    cache._release_lock()
    assert not cache._lock_path().exists()
    cache._release_lock()  # must not raise when there is nothing to remove


_RACE_WORKER_SCRIPT = """
import sys, time
from sisr.cache import LMDBCache

def build(ctx):
    with open(sys.argv[2], "a") as f:
        f.write("build\\n")
    time.sleep(0.3)
    ctx.write_batch([("key_0", b"v0")])

cache = LMDBCache(
    cache_dir=sys.argv[1], name="race", checksum="racecs", length=1,
    map_size=16 * 1024 * 1024, build_fn=build,
    lock_poll_interval=0.02, lock_timeout=5.0,
)
with open(sys.argv[3], "w") as f:
    f.write((cache.get("key_0") or b"").decode())
"""


def test_lmdb_cache_builds_exactly_once_across_two_racing_processes(tmp_path: Path):
    """Two OS processes racing to build the identical (cache_dir, name,
    checksum) -- two concurrent training runs pointed at one cache_dir, the
    scenario this lock exists for -- must call build_fn exactly once; the
    loser waits on the winner's lock and reuses its result.

    Runs genuinely separate interpreters (not threads): LMDB itself refuses a
    second Environment on the same path within one process, so an in-process
    thread-based "race" would trip that same-process restriction rather than
    exercising the actual cross-process lock this guards. Each process gets
    its own result file -- two processes appending to one shared file isn't
    reliably atomic on Windows and would make the harness itself flaky.
    """
    build_log = tmp_path / "build_log.txt"  # only the winner ever writes here
    result_1, result_2 = tmp_path / "result_1.txt", tmp_path / "result_2.txt"
    script = tmp_path / "race_worker.py"
    script.write_text(_RACE_WORKER_SCRIPT)

    p1 = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), str(build_log), str(result_1)]
    )
    time.sleep(0.05)  # give p1 a head start so it wins the lock, not a coin flip
    p2 = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), str(build_log), str(result_2)]
    )
    assert p1.wait(timeout=15) == 0
    assert p2.wait(timeout=15) == 0

    assert build_log.read_text().splitlines() == ["build"], (
        "build_fn must run exactly once across two racing processes"
    )
    assert result_1.read_text() == "v0"
    assert result_2.read_text() == "v0"


def test_acquire_lock_times_out_and_builds_anyway_on_stale_lock(tmp_path: Path):
    """A lock sentinel with no corresponding valid cache (a crashed builder
    that never released it) must not block forever: after lock_timeout
    elapses, the caller proceeds to build rather than waiting indefinitely."""
    name, checksum = "stale", "sc1"
    lock_path = tmp_path / f"{name}_{checksum[:16]}.build.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, b"999999")  # simulates a crashed builder's leftover pid
    os.close(fd)

    call_count = []

    def build(ctx: LMDBCacheBuildContext) -> None:
        call_count.append(1)
        ctx.write_batch([("key_0", b"v0")])

    cache = LMDBCache(
        cache_dir=tmp_path,
        name=name,
        checksum=checksum,
        length=1,
        map_size=_MAP_SIZE,
        build_fn=build,
        lock_poll_interval=0.02,
        lock_timeout=0.1,
    )
    assert call_count == [1], "must build despite the stale lock once the timeout elapses"
    assert cache.get("key_0") == b"v0"
    assert not lock_path.exists(), "taking over from a stale lock should also clean it up"


def test_acquire_lock_polling_returns_false_once_try_load_succeeds(tmp_path: Path):
    """While waiting on a held lock, _try_load succeeding (a concurrent
    builder finished) must end the poll and report "no build needed" --
    without waiting anywhere near the full timeout.

    _try_load itself is exercised elsewhere; this isolates _acquire_lock's
    poll/timeout contract by faking it to report "valid" on its 3rd call,
    rather than racing a second real LMDB build (which LMDB's own
    same-process single-Environment-per-path rule makes unrepresentable with
    threads -- see the cross-process test above for the real end-to-end path).
    """
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="poll",
        checksum="pc1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    lock_path = cache._lock_path()
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)

    calls = []

    def fake_try_load(checksum):
        calls.append(checksum)
        return len(calls) >= 3

    with patch.object(cache, "_try_load", side_effect=fake_try_load):
        acquired = cache._acquire_lock("pc1", poll_interval=0.01, timeout=5.0)

    assert acquired is False
    assert calls == ["pc1", "pc1", "pc1"]
