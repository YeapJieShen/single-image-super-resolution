import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import lmdb
import pytest

from sisr.utils.cache import LMDBCache, LMDBCacheBuildContext

_MAP_SIZE = 16 * 1024 * 1024  # 16 MiB — plenty for tiny test caches


def _make_build_fn(n: int, value_prefix: bytes = b"v"):
    """Returns a build_fn that writes n keys named key_0..key_{n-1}."""

    def build(ctx: LMDBCacheBuildContext) -> None:
        ctx.write_batch([(f"key_{i}", value_prefix + str(i).encode()) for i in range(n)])

    return build


# ---------------------------------------------------------------------------
# Import weight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["sisr.utils", "sisr.utils.cache", "sisr.utils.imresize"])
def test_utils_package_imports_no_torch(module):
    """Nothing reachable from sisr.utils may pull torch in, transitively included.

    parallel_build fans out over a ProcessPoolExecutor, and on spawn platforms
    each worker re-imports the module tree holding its process_fn. A torch
    import reachable from here costs every worker several seconds for a
    dependency the build path never calls. A fresh interpreter is required
    because the test session itself has torch loaded already.

    The package is checked, not only its modules. sisr/utils/__init__.py
    re-exporting anything is what would silently reintroduce the cost, because
    importing one member would then import them all — and sisr.colorspace
    imports torch eagerly, which is exactly why it is not in this package.
    """
    probe = f"import sys, {module}; print('torch' in sys.modules)"
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "False", (
        f"{module} now imports torch (directly or transitively). Every spawned "
        f"LMDB build worker pays that import for nothing — keep torch-dependent code "
        f"out of sisr.utils, and keep its __init__.py free of re-exports."
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


def test_lmdb_try_load_drops_cache_on_genuine_corruption(tmp_path: Path):
    """A corruption-specific exception (bad file header, wrong LMDB version)
    means the directory itself is broken: _try_load returns False and removes
    it so it can be rebuilt. Nothing else can be legitimately using data that
    is genuinely corrupt."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    assert cache.path.exists()
    with patch("sisr.utils.cache.lmdb.open", side_effect=lmdb.CorruptedError("corrupt")):
        assert cache._try_load("abc") is False
    assert not cache.path.exists()


def test_lmdb_try_load_does_not_raise_when_rmtree_fails_on_corruption(tmp_path: Path):
    """Even genuine corruption must not let a failed cleanup (e.g. a file
    still locked by another handle on Windows) escape as an unhandled
    exception -- _try_load must still just report not-ready."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    with (
        patch("sisr.utils.cache.lmdb.open", side_effect=lmdb.CorruptedError("corrupt")),
        patch("sisr.utils.cache.shutil.rmtree", side_effect=PermissionError("locked")),
    ):
        assert cache._try_load("abc") is False


def test_lmdb_try_load_does_not_delete_cache_on_same_process_reopen_error(tmp_path: Path):
    """A second lmdb.open of an already-open environment in the same process
    raises a bare lmdb.Error ('environment already open in this process'),
    not a corruption-specific subclass -- empirically hit when two datasets
    in one process both open caches with lock=False. This is environmental,
    not corruption: the directory must survive and _try_load must just
    report not-ready so a later retry can succeed."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    cache.get_env()  # holds an open Environment at cache.path in this process

    assert cache._try_load("abc") is False
    assert cache.path.exists(), "a same-process reopen conflict is not corruption"


def test_lmdb_try_load_does_not_delete_cache_on_permission_error(tmp_path: Path):
    """A PermissionError opening the environment (e.g. Windows denying access
    to a file another process still has mapped) is environmental too, not
    corruption -- must not delete the directory."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    with patch("sisr.utils.cache.lmdb.open", side_effect=PermissionError("denied")):
        assert cache._try_load("abc") is False
    assert cache.path.exists()


def test_lmdb_try_load_does_not_delete_cache_on_non_corruption_lmdb_error(tmp_path: Path):
    """Only the specific corruption subclasses (CorruptedError, InvalidError,
    VersionMismatchError) justify deleting the directory. Other lmdb.Error
    subclasses (e.g. a full lock table) are environmental and must not delete."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    with patch("sisr.utils.cache.lmdb.open", side_effect=lmdb.LockError("lock table full")):
        assert cache._try_load("abc") is False
    assert cache.path.exists()


def test_lmdb_try_load_treats_missing_length_as_stale_not_crash(tmp_path: Path):
    """A cache whose __checksum__ matches but __length__ is absent (e.g. a
    hand-built or foreign-layout cache) must be treated as incomplete/stale,
    not crash with AttributeError from int(None.decode())."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="test",
        checksum="abc",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    env = lmdb.open(str(cache.path), map_size=_MAP_SIZE)
    with env.begin(write=True) as txn:
        txn.delete(b"__length__")
    env.close()

    assert cache._try_load("abc") is False
    assert cache.path.exists(), "an incomplete cache is stale, not corrupt -- must not be deleted"


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
    with patch("sisr.utils.cache.lmdb.open", side_effect=ValueError("unexpected")):
        with pytest.raises(ValueError, match="unexpected"):
            cache._try_load("abc")


def test_construction_raises_instead_of_silently_rebuilding_on_same_process_conflict(
    tmp_path: Path,
):
    """Constructing a *second* LMDBCache at an existing, valid path while a
    first instance's env is still open in this process must not silently
    treat the resulting 'already open' error as 'rebuild me'.

    Previously: environmental open failure -> _try_load returns False ->
    __init__ proceeds as if nothing were there -> the sentinel is free -> a
    full, destructive rebuild runs (potentially minutes for a real dataset)
    of a perfectly valid cache, ending in _build/_publish trying to touch a
    directory this same process still has mapped. It must instead raise
    immediately, chaining the original lmdb error, and touch neither the
    lock nor the existing cache at all.
    """
    call_count = [0]

    def counting_build(ctx: LMDBCacheBuildContext) -> None:
        call_count[0] += 1
        ctx.write_batch([("key_0", b"v0")])

    first = LMDBCache(
        cache_dir=tmp_path,
        name="shared",
        checksum="c1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=counting_build,
    )
    first.get_env()  # holds this process's only allowed handle open at this path

    with pytest.raises(RuntimeError, match="already open"):
        LMDBCache(
            cache_dir=tmp_path,
            name="shared",
            checksum="c1",
            length=1,
            map_size=_MAP_SIZE,
            build_fn=counting_build,
        )

    assert call_count[0] == 1, (
        "must not silently rebuild a valid cache blocked by a same-process handle"
    )
    assert first.get("key_0") == b"v0", "the first instance's cache must remain untouched and valid"


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

    with patch("sisr.utils.cache.ProcessPoolExecutor") as mock_pool:
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

    with patch("sisr.utils.cache.ProcessPoolExecutor") as mock_pool:
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
    cache._release_lock()  # _owns_lock is already False here -- a no-op, not a re-check


def test_release_lock_does_not_remove_a_sentinel_it_never_created(tmp_path: Path):
    """_release_lock must be a no-op for an instance that never won (or
    stole) the build lock -- otherwise any call to it (e.g. a defensive
    double-call, or reuse of a cache object across a retry) could delete a
    live builder's sentinel out from under it, letting a third process
    immediately win a fresh race into the same directory."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="own",
        checksum="c",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    # cache already built + released its own lock inside __init__ above.
    lock_path = cache._lock_path()
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, b"424242")  # simulates another (live) process's sentinel
    os.close(fd)

    cache._release_lock()

    assert lock_path.exists(), "must not delete a sentinel this instance never created"


_RACE_WORKER_SCRIPT = """
import sys, time
from sisr.utils.cache import LMDBCache

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


_LIVE_BUILDER_WORKER_SCRIPT = """
import sys, time
from sisr.utils.cache import LMDBCache

def build(ctx):
    with open(sys.argv[3], "a") as f:
        f.write("build\\n")
    time.sleep(float(sys.argv[4]))
    ctx.write_batch([("key_0", b"v0")])

cache = LMDBCache(
    cache_dir=sys.argv[1], name="live", checksum="livecs", length=1,
    map_size=16 * 1024 * 1024, build_fn=build,
    lock_poll_interval=0.02, lock_timeout=float(sys.argv[5]),
)
with open(sys.argv[2], "w") as f:
    f.write((cache.get("key_0") or b"").decode())
"""


def test_live_builder_past_lock_timeout_is_never_preempted(tmp_path: Path):
    """Process A wins the lock and is still inside a slow build_fn when
    process B's much shorter lock_timeout elapses. B must not treat that as
    an abandoned lock: A's pid is still alive, so B must wait rather than
    rebuild -- the old fall-through-on-timeout behaviour instead had B call
    _build, which opened by shutil.rmtree-ing A's live, memory-mapped LMDB
    directory (a PermissionError crash on Windows, silent lost writes on
    Linux).

    The build-log assertion is the platform-independent half of this test:
    on POSIX, the old buggy code's in-place rmtree of a live, open directory
    often *succeeds* (unlike Windows' PermissionError), so B would silently
    rebuild -- every process-exit-code/content assertion could pass even
    though build_fn ran twice and A's own write was lost to an orphaned,
    unlinked inode, because B's redundant rebuild happens to reconstruct the
    identical value. A shared log line only the executing build_fn appends
    to catches that regardless of platform: it must show exactly one build.
    """
    result_a, result_b = tmp_path / "result_a.txt", tmp_path / "result_b.txt"
    build_log = tmp_path / "build_log.txt"
    script = tmp_path / "live_worker.py"
    script.write_text(_LIVE_BUILDER_WORKER_SCRIPT)
    cache_path = tmp_path / "live_livecs"
    lock_path = tmp_path / "live_livecs.build.lock"

    # A: wins the lock immediately, then sleeps 1.5s inside build_fn --
    # comfortably longer than B's lock_timeout below.
    proc_a = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), str(result_a), str(build_log), "1.5", "30"]
    )
    # A's cold interpreter + import (torch-free, but tqdm's asyncio import
    # chain alone measures ~0.4s) must finish and claim the sentinel before
    # this check -- 1.0s gives comfortable margin over that.
    time.sleep(1.0)
    assert lock_path.exists(), "A must have claimed the sentinel before B ever starts"
    assert lock_path.read_text() == str(proc_a.pid), (
        "A must be the one holding the lock -- otherwise this test isn't exercising "
        "the timeout-while-holder-alive path at all"
    )

    # B: loses the race, and its own lock_timeout elapses long before A's
    # build_fn returns. 1.0s (not e.g. 0.2s) keeps its 3x-lock_timeout hard
    # cap at 3.0s -- 0.2s left only ~0.6s of margin for B to observe A's
    # publish, which cold-vs-warm interpreter start asymmetry (A pays a
    # cold import, B's is warmer) plus tens of ms of Windows publish time
    # could burn through, flaking B into a TimeoutError on a loaded CI box.
    proc_b = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), str(result_b), str(build_log), "0", "1.0"]
    )

    assert proc_a.wait(timeout=15) == 0, "A's build must complete without a crash from B"
    assert proc_b.wait(timeout=15) == 0, (
        "B must not crash trying to destroy A's live cache after its lock_timeout elapses"
    )
    assert cache_path.exists(), "A's cache directory must survive B's timed-out wait"
    assert build_log.read_text().splitlines() == ["build"], (
        "build_fn must run exactly once -- B must never rebuild A's live cache, on any platform"
    )
    assert result_a.read_text() == "v0"
    assert result_b.read_text() == "v0"


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


def test_acquire_lock_treats_own_recycled_pid_as_dead_not_alive(tmp_path: Path):
    """A sentinel recording THIS process's own pid must not be treated as a
    live holder -- it can only mean a crashed builder's pid got recycled to
    this very process (Windows in particular reuses pids in small
    multiples, so crash-then-restart-with-the-same-pid is realistic, not
    theoretical). Otherwise this process would wait on "itself" forever.
    Once stale, it must be treated as abandoned and taken over instead."""
    name, checksum = "selfpid", "sp1"
    lock_path = tmp_path / f"{name}_{checksum[:16]}.build.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())  # "our own" pid, as if recycled from a crashed run
    os.close(fd)
    stale_mtime = time.time() - 10
    os.utime(lock_path, (stale_mtime, stale_mtime))

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
    assert call_count == [1], "a sentinel bearing this process's own pid must not block it forever"
    assert cache.get("key_0") == b"v0"


def test_acquire_lock_raises_timeout_error_on_confirmed_live_holder_past_hard_cap(tmp_path: Path):
    """Waiting on a confirmed-*alive* holder must still be bounded: past
    3x lock_timeout with no sign of it finishing, this must raise
    TimeoutError instead of waiting forever -- an operator needs a way to
    notice and intervene."""
    name, checksum = "hardcap", "hc1"
    lock_path = tmp_path / f"{name}_{checksum[:16]}.build.lock"
    other_pid = os.getppid()  # genuinely alive, and guaranteed not to equal our own pid
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(other_pid).encode())
    os.close(fd)

    with pytest.raises(TimeoutError, match="Refusing to wait indefinitely"):
        LMDBCache(
            cache_dir=tmp_path,
            name=name,
            checksum=checksum,
            length=1,
            map_size=_MAP_SIZE,
            build_fn=_make_build_fn(1),
            lock_poll_interval=0.02,
            lock_timeout=0.05,
        )


def test_acquire_lock_toctou_reread_skips_unlink_if_pid_changed(tmp_path: Path):
    """Immediately before unlinking a dead+stale sentinel to take over, the
    pid is re-read; if it changed (another waiter already reclaimed a fresh
    sentinel in the interim), the stale sentinel must be left alone rather
    than deleting what might now be a live claim."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="toctou",
        checksum="tc1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    lock_path = cache._lock_path()
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, b"999999")  # a dead pid
    os.close(fd)
    stale_mtime = time.time() - 100
    os.utime(lock_path, (stale_mtime, stale_mtime))

    read_calls = []

    def fake_read_lock_pid():
        read_calls.append(1)
        # 1st call (the outer holder_pid read): the dead pid above. 2nd call
        # (the TOCTOU re-read right before unlinking): a DIFFERENT pid, as
        # if another waiter had just reclaimed the sentinel in between.
        return 999999 if len(read_calls) == 1 else 888888

    try_load_calls = []

    def fake_try_load(checksum):
        try_load_calls.append(1)
        return len(try_load_calls) >= 2  # succeed on the 2nd poll, ending the loop cleanly

    with (
        patch.object(cache, "_read_lock_pid", side_effect=fake_read_lock_pid),
        patch.object(cache, "_try_load", side_effect=fake_try_load),
    ):
        acquired = cache._acquire_lock("tc1", poll_interval=0.01, timeout=0.05)

    assert acquired is False, "must report no build needed once _try_load succeeds"
    assert lock_path.read_text() == "999999", (
        "the original sentinel must survive a mismatched TOCTOU re-read -- it must never be "
        "unlinked when the pid changed between the two reads"
    )


def test_release_lock_does_not_remove_sentinel_if_content_was_replaced(tmp_path: Path):
    """If the on-disk sentinel no longer records this process's own pid --
    e.g. the rare TOCTOU window _acquire_lock's takeover narrows but cannot
    fully close -- _release_lock must not delete it even though this
    instance's _owns_lock flag is still True from when it originally
    claimed the lock. The ownership flag alone is not sufficient; the
    content is re-checked too."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="clobber",
        checksum="cl1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    cache._acquire_lock("cl1", poll_interval=0.01, timeout=0.05)
    assert cache._owns_lock is True
    # Simulate another process's takeover clobbering our sentinel's content
    # in between.
    cache._lock_path().write_text("999999")

    cache._release_lock()

    assert cache._lock_path().read_text() == "999999", (
        "must not delete a sentinel whose content no longer matches this process's own pid"
    )


def test_heartbeat_survives_transient_os_error_and_keeps_refreshing(tmp_path: Path):
    """The heartbeat thread must not give up after the first transient
    OSError touching the sentinel (e.g. a momentary Windows sharing
    violation) -- only a FileNotFoundError (the sentinel is genuinely gone)
    should stop it. Simulate one transient failure, then let real os.utime
    calls through, and confirm the thread kept retrying rather than exiting
    silently on the first hiccup."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="hb",
        checksum="h1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    cache._acquire_lock("h1", poll_interval=0.01, timeout=0.05)

    real_utime = os.utime
    calls = []

    def flaky_utime(path, times):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("transient")
        real_utime(path, times)

    with patch("sisr.utils.cache.os.utime", side_effect=flaky_utime):
        cache._start_heartbeat(timeout=0.05)  # interval clamps to _HEARTBEAT_MIN_INTERVAL (0.01s)
        time.sleep(0.3)
        cache._stop_heartbeat()

    assert len(calls) >= 3, "the heartbeat must keep retrying past the first transient failure"


def test_construction_releases_lock_when_heartbeat_thread_fails_to_start(tmp_path: Path):
    """If Thread.start() itself raises (e.g. the OS refuses a new thread),
    _stop_heartbeat's join() must not blow up on a thread that was
    constructed but never started -- that would mask the original start()
    error and skip _release_lock on the next line (both run inside the same
    finally), leaving a live-pid sentinel that stalls every other waiter for
    up to 3x lock_timeout. The original error must still surface, and the
    sentinel must still be released."""
    with patch(
        "sisr.utils.cache.threading.Thread.start",
        side_effect=RuntimeError("can't start new thread"),
    ):
        with pytest.raises(RuntimeError, match="can't start new thread"):
            LMDBCache(
                cache_dir=tmp_path,
                name="hbfail",
                checksum="hf1",
                length=1,
                map_size=_MAP_SIZE,
                build_fn=_make_build_fn(1),
            )

    lock_path = tmp_path / "hbfail_hf1.build.lock"
    assert not lock_path.exists(), (
        "the sentinel must be released even though the heartbeat thread never started"
    )


# ---------------------------------------------------------------------------
# Windows pid liveness (ACCESS_DENIED vs INVALID_PARAMETER)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific liveness path")
def test_pid_alive_windows_access_denied_means_alive():
    """OpenProcess failing with ERROR_ACCESS_DENIED means the process exists
    (e.g. owned by another user/session) -- it must be treated as alive, or
    a live builder running under a different account could be preempted."""
    from sisr.utils.cache import _pid_alive_windows  # local: doesn't exist pre-fix

    fake_kernel32 = MagicMock()
    fake_kernel32.OpenProcess.return_value = 0
    with (
        patch("sisr.utils.cache._kernel32", fake_kernel32),
        patch("sisr.utils.cache.ctypes.get_last_error", return_value=5),  # ERROR_ACCESS_DENIED
    ):
        assert _pid_alive_windows(4242) is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific liveness path")
def test_pid_alive_windows_invalid_parameter_means_dead():
    """OpenProcess failing with ERROR_INVALID_PARAMETER (empirically what a
    genuinely nonexistent pid produces) means no such process."""
    from sisr.utils.cache import _pid_alive_windows  # local: doesn't exist pre-fix

    fake_kernel32 = MagicMock()
    fake_kernel32.OpenProcess.return_value = 0
    with (
        patch("sisr.utils.cache._kernel32", fake_kernel32),
        patch("sisr.utils.cache.ctypes.get_last_error", return_value=87),  # ERROR_INVALID_PARAMETER
    ):
        assert _pid_alive_windows(4242) is False


# ---------------------------------------------------------------------------
# Publish (temp-sibling build + atomic rename) and crash-safety sweep
# ---------------------------------------------------------------------------


def test_publish_does_not_destroy_a_still_open_target_directory(tmp_path: Path):
    """_publish must move a pre-existing target aside via ``os.replace`` (a
    rename) rather than ``rmtree``-ing it in place: a single rename is
    atomic and can never leave the directory half-deleted, unlike an
    interrupted ``rmtree`` that might remove some files before hitting one
    it can't (e.g. still memory-mapped by a reader in another process on
    Windows -- verified empirically: even a *rename* of such a directory can
    itself fail there, which is exactly why this must never be an unlink).
    The only ``rmtree`` this method may still perform is the final,
    best-effort cleanup of the *renamed-away* trash sibling -- never of the
    live target itself.
    """
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="pub",
        checksum="old",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1, value_prefix=b"old"),
    )
    live_target = cache.path

    # Build a replacement directly, bypassing __init__ (which would now
    # raise on a same-process handle if one were open) -- this isolates
    # _publish's own move-aside mechanism.
    tmp_build_dir = cache.path.with_name(f"{cache.path.name}.build.{os.getpid()}.tmp")
    tmp_env = lmdb.open(str(tmp_build_dir), map_size=_MAP_SIZE)
    with tmp_env.begin(write=True) as txn:
        txn.put(b"key_0", b"new0")
        txn.put(b"__checksum__", b"old")
        txn.put(b"__length__", b"1")
    tmp_env.close()

    with patch("sisr.utils.cache.shutil.rmtree") as mock_rmtree:
        cache._publish(tmp_build_dir)

    assert mock_rmtree.call_count == 1, (
        "the only rmtree call must be the trash sibling's final cleanup"
    )
    (swept_path,), _ = mock_rmtree.call_args
    assert swept_path != live_target, (
        "the live target itself must never be rmtree'd, only renamed aside"
    )

    fresh_env = lmdb.open(str(cache.path), readonly=True, lock=False)
    with fresh_env.begin() as txn:
        assert bytes(txn.get(b"key_0")) == b"new0", (
            "the new build must now be live at the target path"
        )
    fresh_env.close()


def test_sweep_stale_siblings_removes_only_dead_pid_temp_dirs(tmp_path: Path):
    """A leftover .build.<pid>.tmp or .trash.<pid>.tmp from a crashed build
    whose pid is no longer alive must be swept -- these are full,
    map_size-presized LMDB directories (several GB for a real dataset), so a
    crash before cleanup otherwise leaks that disk space forever. A sibling
    belonging to this process's own (obviously alive) pid must survive."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="sweep",
        checksum="sw1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )

    dead_build_leftover = cache.path.with_name(f"{cache.path.name}.build.999999.tmp")
    dead_build_leftover.mkdir()
    dead_trash_leftover = cache.path.with_name(f"{cache.path.name}.trash.999998.tmp")
    dead_trash_leftover.mkdir()
    own_leftover = cache.path.with_name(f"{cache.path.name}.build.{os.getpid()}.tmp")
    own_leftover.mkdir()

    cache._sweep_stale_siblings()

    assert not dead_build_leftover.exists(), "a dead pid's leftover build temp must be swept"
    assert not dead_trash_leftover.exists(), "a dead pid's leftover trash temp must be swept"
    assert own_leftover.exists(), "this process's own pid must never be swept as 'dead'"


def test_build_invokes_sibling_sweep_automatically(tmp_path: Path):
    """_build must call the sweep itself as part of a real rebuild, not just
    have it available for manual use -- a leftover from a previous crash
    must be gone after the very next build that touches this cache."""
    cache = LMDBCache(
        cache_dir=tmp_path,
        name="autosweep",
        checksum="v1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )
    leftover = cache.path.with_name(f"{cache.path.name}.build.999999.tmp")
    leftover.mkdir()
    shutil.rmtree(cache.path)  # force a genuine rebuild at the identical (name, checksum) path

    LMDBCache(
        cache_dir=tmp_path,
        name="autosweep",
        checksum="v1",
        length=1,
        map_size=_MAP_SIZE,
        build_fn=_make_build_fn(1),
    )

    assert not leftover.exists()
