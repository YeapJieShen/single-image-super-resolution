"""Checksum-validated LMDB key-value store with parallel build support.

:class:`LMDBCache` is used by both :class:`~sisr.datasets.srcnn.TrainDataset`
and :class:`~sisr.datasets.srresnet.TrainDataset` to persist whole decoded HR
images, raw uint8; each derives its LR (and, for SRCNN, its deterministic
sub-image grid) at read time via :meth:`LMDBCache.get_buffer`.

This module deliberately imports no torch, and must stay that way.
:meth:`LMDBCacheBuildContext.parallel_build` fans work out over a
``ProcessPoolExecutor``, and on spawn platforms (Windows, macOS) every worker
re-imports the module tree holding its ``process_fn`` — so a torch import
reachable from here costs each worker several seconds for a dependency the
build path never calls. ``tests/test_cache.py`` asserts the module stays
torch-free; colorspace math lives in :mod:`sisr.colorspace` for this reason.
"""

import ctypes
import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import lmdb
from tqdm.auto import tqdm

_logger = logging.getLogger(__name__)

# Only these mean the LMDB directory itself is broken (bad header, wrong
# on-disk version, ...) and safe to delete. Everything else -- notably the
# bare lmdb.Error raised for "environment already open in this process", or
# lmdb.LockError/DiskError -- is environmental and must not delete anything.
_CORRUPTION_ERRORS = (lmdb.CorruptedError, lmdb.InvalidError, lmdb.VersionMismatchError)

_HEARTBEAT_MIN_INTERVAL = 0.01
_HEARTBEAT_MAX_INTERVAL = 30.0

# How many multiples of lock_timeout to wait on a confirmed-live holder
# before giving up and raising -- bounds an otherwise-unbounded wait.
_LIVE_HOLDER_WAIT_MULTIPLE = 3

# Windows liveness check: OpenProcess's failure mode must be inspected via
# GetLastError, which ctypes only tracks reliably through a use_last_error=True
# handle (the default cached ctypes.windll.kernel32 does not guarantee the
# value survives ctypes' own bookkeeping between the call and the check).
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if sys.platform == "win32" else None


def _pid_alive_windows(pid: int) -> bool:
    """Windows liveness check for :func:`_pid_alive`, split out for testing.

    ``OpenProcess`` failing does not always mean "no such process": a
    builder running under another user account or session is a real,
    live process this one simply lacks permission to query
    (``ERROR_ACCESS_DENIED``) and must not be treated as dead. Only an
    actually-missing pid reports something else (``ERROR_INVALID_PARAMETER``
    in practice).

    Args:
        pid: Process id to check.

    Returns:
        ``True`` if *pid* is a running process, or one this process is
        merely not permitted to query.
    """
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5

    assert _kernel32 is not None  # only called when sys.platform == "win32"
    handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        exit_code = ctypes.c_ulong()
        if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        _kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    """Best-effort, portable check for whether *pid* names a running process.

    ``os.kill(pid, 0)`` is the standard POSIX liveness idiom, but on Windows
    CPython's ``os.kill`` maps arbitrary signal numbers (including 0) onto
    ``TerminateProcess`` -- calling it here could actually kill a live
    builder rather than merely check it. So on that platform this queries
    the process table via :func:`_pid_alive_windows` instead.

    Args:
        pid: Process id to check.

    Returns:
        ``True`` if a process with this pid is currently running.
    """
    if sys.platform == "win32":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    return True


class LMDBCacheBuildContext:
    """Context object passed to the ``build_fn`` callback of :class:`LMDBCache`.

    Provides helpers for writing data into the LMDB being built.

    Args:
        env: The open, writable LMDB environment.
        use_tqdm: Whether to display progress bars.
    """

    def __init__(self, env: lmdb.Environment, use_tqdm: bool = False):
        self.env = env
        self.use_tqdm = use_tqdm

    def write_batch(self, pairs: Sequence[tuple[str, bytes]]) -> None:
        """Writes a batch of key-value pairs in a single transaction.

        Args:
            pairs: Sequence of ``(key, value)`` tuples to write.
        """
        txn = self.env.begin(write=True)
        for key, value in pairs:
            txn.put(key.encode(), value)
        txn.commit()

    def parallel_build(
        self,
        items: Sequence[Any],
        process_fn: Callable[..., list[tuple[str, bytes]]],
        process_args: Sequence[Sequence[Any]] | None = None,
        num_workers: int | None = None,
        desc: str = "Building LMDB cache",
    ) -> None:
        """Processes *items* and writes the results to LMDB.

        *process_fn* must be a top-level (picklable) callable returning a
        list of ``(key, value_bytes)`` tuples. With more than one effective
        worker a ``ProcessPoolExecutor`` runs a sliding window of
        *num_workers* in-flight jobs while the main process writes completed
        results; with ``<= 1`` effective worker the jobs run inline (no
        subprocess).

        Args:
            items: One item per job (e.g. a list of image paths).
            process_fn: Top-level callable invoked per item as
                ``(item, *extra_args)``, returning its keyed pairs.
            process_args: Per-item extra arguments for *process_fn*, unpacked
                as positional args. ``None`` calls ``process_fn(item)`` with
                no extras.
            num_workers: Maximum parallel workers. ``None`` resolves to
                ``min(os.cpu_count() or 1, len(items))``; an effective count
                ``<= 1`` runs inline with no ``ProcessPoolExecutor``.
            desc: Description shown on the ``tqdm`` progress bar.
        """
        n_items = len(items)
        if num_workers is None:
            num_workers = min(os.cpu_count() or 1, n_items)
        num_workers = min(num_workers, n_items)

        pbar = tqdm(total=n_items, desc=desc, unit="item") if self.use_tqdm else None

        # <= 1 effective worker: skip the pool. Its spawn/import cost, plus the
        # hazard of nesting one inside a test/xdist worker, isn't worth it here.
        if num_workers <= 1:
            for i in range(n_items):
                args = (items[i],)
                if process_args is not None:
                    args = args + tuple(process_args[i])
                self.write_batch(process_fn(*args))
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
            return

        next_submit = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pending: set[Future] = set()

            def _submit() -> None:
                nonlocal next_submit
                if next_submit < n_items:
                    args = (items[next_submit],)
                    if process_args is not None:
                        args = args + tuple(process_args[next_submit])
                    pending.add(executor.submit(process_fn, *args))
                    next_submit += 1

            for _ in range(num_workers):
                _submit()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.discard(future)
                    self.write_batch(future.result())
                    if pbar is not None:
                        pbar.update(1)
                    _submit()

        if pbar is not None:
            pbar.close()


class LMDBCache:
    """A checksum-validated LMDB key-value store with parallel build support.

    On construction the cache checks whether a valid LMDB database
    already exists (matched by *checksum*). If not, it calls ``build_fn``
    to populate the database from scratch using a pool of worker processes.

    After construction the cache is read-only. Call :meth:`get` to
    retrieve individual entries, or :meth:`get_env` for direct LMDB access.

    Two independent processes building the same cache (same *cache_dir*,
    *name*, *checksum*) race by design — LMDB itself already serialises
    writers and ``__checksum__`` is written last, so a half-built cache is
    never mistaken for a complete one. An advisory file lock (see
    :meth:`_acquire_lock`) exists only to *avoid duplicate work*, not to
    provide correctness: a losing process polls for the winner's result and
    joins it. The lock's holder is never preempted while it is provably
    alive (its pid running — and not merely this waiting process's own pid,
    which a crashed holder's can be recycled to — corroborated by a
    heartbeat that refreshes the sentinel's mtime during the build) — a
    losing process instead keeps waiting past *lock_timeout*, re-logging
    periodically, rather than racing a slow but healthy build. Only a
    holder whose pid has died (or is our own) *and* whose sentinel has gone
    stale for longer than *lock_timeout* is treated as abandoned and taken
    over. Waiting on a *live* holder is itself capped: past
    ``3 * lock_timeout`` this raises ``TimeoutError`` naming the sentinel
    and the manual remedy, rather than blocking forever with no way for an
    operator to notice. Waiting forever on a genuinely stale lock is the
    other failure mode this must never have; rebuilding a cache that
    already exists is merely wasted time.

    Args:
        cache_dir: Parent directory for the LMDB database.
        name: Prefix used in the LMDB folder name (e.g. ``'hr_raw'``).
        checksum: Hex digest identifying the current configuration; a
            mismatch triggers a rebuild.
        length: Total number of entries that will be stored.
        map_size: Maximum size of the LMDB database in bytes.
        metadata: Extra key-value pairs to persist alongside the data.
        build_fn: Populates the database, receiving a single
            :class:`LMDBCacheBuildContext`. ``None`` with no valid cache
            found raises ``RuntimeError``.
        use_tqdm: Whether to display a progress bar during the build.
        lock_poll_interval: Seconds between checks for a concurrent
            builder's result while waiting on its lock.
        lock_timeout: Seconds to wait on a held lock, past which a *dead*
            holder's lock is considered abandoned. A live holder is never
            preempted regardless of this value.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        name: str,
        checksum: str,
        length: int,
        map_size: int,
        metadata: dict[str, str] | None = None,
        build_fn: Callable[[LMDBCacheBuildContext], None] | None = None,
        use_tqdm: bool = False,
        lock_poll_interval: float = 0.5,
        lock_timeout: float = 600.0,
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lmdb_path = self._cache_dir / f"{name}_{checksum[:16]}"
        self._length = length
        self._metadata = metadata or {}
        self._env: lmdb.Environment | None = None
        self._owns_lock = False
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop: threading.Event | None = None

        if not self._try_load(checksum, strict=True):
            if build_fn is None:
                raise RuntimeError(
                    f"No valid LMDB cache found at {self._lmdb_path} and no build_fn was provided."
                )
            if self._acquire_lock(checksum, lock_poll_interval, lock_timeout):
                try:
                    # _start_heartbeat lives inside this try too: if starting the
                    # thread itself fails, the sentinel must still be released in
                    # the finally below rather than leaked until it goes stale.
                    self._start_heartbeat(lock_timeout)
                    # Re-check: a concurrent builder may have finished (and released
                    # its lock) between our first _try_load and winning this one.
                    if not self._try_load(checksum, strict=True):
                        self._build(checksum, length, map_size, build_fn, use_tqdm)
                finally:
                    self._stop_heartbeat()
                    self._release_lock()

    @property
    def path(self) -> Path:
        """Path to the LMDB database directory."""
        return self._lmdb_path

    @property
    def length(self) -> int:
        """Number of entries stored in the cache."""
        return self._length

    def get_env(self) -> lmdb.Environment:
        """Returns the LMDB environment, opening it lazily on first call.

        Each ``DataLoader`` worker process must call this independently:
        LMDB memory-mapped environments cannot be shared across processes
        created via ``spawn``.
        """
        if self._env is None:
            self._env = lmdb.open(str(self._lmdb_path), readonly=True, lock=False)
        return self._env

    def get(self, key: str) -> bytes | None:
        """Reads a single value from the cache.

        Args:
            key: The string key to look up.

        Returns:
            The raw bytes stored under *key*, or ``None`` if absent.
        """
        env = self.get_env()
        with env.begin(write=False, buffers=True) as txn:
            buf = txn.get(key.encode())
            if buf is None:
                return None
            return bytes(buf)

    @contextmanager
    def get_buffer(self, key: str) -> Iterator[memoryview | None]:
        """Yields a zero-copy view onto the raw value stored at *key*.

        The read transaction is opened on entry and committed on exit, so the
        yielded ``memoryview`` aliases memory that is **only valid inside this
        ``with`` block** — LMDB is free to reclaim it the moment the
        transaction closes. Do all interpretation (``np.frombuffer``, slicing)
        and copying (``.copy()``, ``torch.from_numpy(...).clone()``) *before*
        the block exits. This is a context manager rather than a plain return
        specifically so a caller cannot accidentally retain the view past its
        transaction — there is no variable holding the buffer until you
        actually enter the ``with``.

        Args:
            key: The string key to look up.

        Yields:
            A ``memoryview`` onto the raw value bytes, or ``None`` if *key* is
            absent.
        """
        env = self.get_env()
        with env.begin(write=False, buffers=True) as txn:
            yield txn.get(key.encode())

    def get_batch(self, keys: Sequence[str]) -> list[bytes | None]:
        """Reads multiple values from the cache in a single transaction.

        Args:
            keys: Sequence of string keys to look up.

        Returns:
            A list of raw bytes (or ``None`` for missing keys), in the same
            order as *keys*.
        """
        env = self.get_env()
        results = []
        with env.begin(write=False, buffers=True) as txn:
            for key in keys:
                buf = txn.get(key.encode())
                results.append(bytes(buf) if buf is not None else None)
        return results

    def _try_load(self, checksum: str, strict: bool = False) -> bool:
        """Validates an existing LMDB against *checksum*; ``True`` if ready to use.

        Only a corruption-specific failure to *open* the environment (bad
        file header, wrong on-disk version — see :data:`_CORRUPTION_ERRORS`)
        is presumed genuine corruption, dropping the directory so a later
        call can rebuild it. Every other failure to open is environmental --
        notably the bare ``lmdb.Error`` ("already open in this process")
        raised when this same process holds another handle on the same path,
        or a Windows failure while a concurrent writer *in another process*
        is still active -- and never means the directory is corrupt.

        Args:
            checksum: Hex digest the stored ``__checksum__`` must match.
            strict: When ``True``, an environmental open failure raises
                instead of returning ``False``. Used at the two call sites
                where this process is about to become the builder: silently
                treating "another handle is already open" as "rebuild me"
                previously caused a full, destructive rebuild of a perfectly
                valid cache. Left ``False`` (the default) while merely
                polling on someone else's lock, where the identical failure
                is often a transient, self-resolving artifact of a
                concurrent writer in another process.

        Returns:
            ``True`` if the cache is present, checksum-matched, and ready.

        Raises:
            RuntimeError: If *strict* and the environment could not be
                opened for a non-corruption reason.
        """
        if not self._lmdb_path.exists():
            return False
        try:
            env = lmdb.open(str(self._lmdb_path), readonly=True, lock=False)
        except _CORRUPTION_ERRORS:
            # Even this cleanup is guarded: a failure here (e.g. a lingering
            # open handle) must not escape as an unhandled exception either.
            try:
                if self._lmdb_path.exists():
                    shutil.rmtree(self._lmdb_path)
            except OSError:
                pass
            return False
        except (lmdb.Error, OSError) as exc:
            if strict:
                raise RuntimeError(
                    f"Cannot open the existing cache at {self._lmdb_path}: {exc}. This is "
                    "not corruption: most likely another handle to this cache is already "
                    "open in this process (close it first) and a rebuild would only race "
                    "and destroy it, or the OS-level conflict needs investigating. Refusing "
                    "to silently rebuild."
                ) from exc
            return False

        try:
            with env.begin(write=False) as txn:
                stored = txn.get(b"__checksum__")
                if stored is None or stored.decode() != checksum:
                    return False
                length_raw = txn.get(b"__length__")
                if length_raw is None:
                    return False
                try:
                    self._length = int(length_raw.decode())
                except ValueError:
                    return False
            return True
        except (lmdb.Error, OSError):
            return False
        finally:
            env.close()

    def _lock_path(self) -> Path:
        """Sentinel path for the advisory build lock, scoped to this checksum."""
        return self._lmdb_path.with_name(self._lmdb_path.name + ".build.lock")

    def _claim_sentinel(self) -> bool:
        """Creates this process's lock sentinel; ``True`` on success.

        ``False`` means the sentinel already exists (someone else holds, or
        just won, the race).
        """
        try:
            fd = os.open(str(self._lock_path()), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)  # always close, even if the write itself failed
        self._owns_lock = True
        return True

    def _read_lock_pid(self) -> int | None:
        """Returns the pid recorded in the lock sentinel, or ``None`` if unreadable."""
        try:
            raw = self._lock_path().read_text()
        except OSError:
            return None
        try:
            return int(raw.strip())
        except ValueError:
            return None

    def _lock_is_stale(self, timeout: float) -> bool:
        """``True`` if the sentinel's mtime hasn't been refreshed within *timeout*."""
        try:
            mtime = self._lock_path().stat().st_mtime
        except OSError:
            return True  # sentinel vanished entirely -- nothing left to hold onto
        return (time.time() - mtime) > timeout

    def _acquire_lock(self, checksum: str, poll_interval: float, timeout: float) -> bool:
        """Becomes the sole builder for *checksum*, or waits on whoever already is.

        A live holder — its pid still running, and *not this process's own
        pid* (a crashed builder's pid can be recycled to this very process;
        Windows in particular reuses pids in small multiples, making
        crash-then-restart-with-the-same-pid a real scenario, not a
        theoretical one) — is never preempted: this waits past *timeout*,
        re-logging periodically so a long wait stays diagnosable, rather
        than racing a slow but healthy build. Takeover only happens once the
        holder's pid is confirmed dead (or is our own) *and* its sentinel's
        mtime (refreshed by the holder's heartbeat while it builds, see
        :meth:`_start_heartbeat`) has gone stale for longer than *timeout*
        — i.e. it crashed without releasing the lock. Waiting on a
        confirmed-*live* holder is itself capped at
        ``_LIVE_HOLDER_WAIT_MULTIPLE * timeout``: past that this raises
        rather than blocking forever, so a genuinely stuck wait surfaces
        instead of hanging silently.

        Note on filesystem assumptions: staleness is judged by the
        sentinel's mtime against wall-clock ``time.time()``, so it assumes
        all racing processes share a consistent clock and a filesystem with
        working mtimes (true for a local disk; a clock-skewed network
        filesystem, or an NTP step during the wait, could misjudge
        staleness in either direction). This lock is advisory and best-effort
        by design (see the class docstring) — that assumption is accepted,
        not solved, here.

        Returns:
            ``True`` if the caller should proceed to build — either it won
            the lock outright, or it took over one abandoned by a dead
            holder. ``False`` if a concurrent builder produced a valid cache
            while waiting — :meth:`_try_load` has already refreshed
            :attr:`_length` in that case.

        Raises:
            TimeoutError: If a confirmed-live holder still has not finished
                after ``_LIVE_HOLDER_WAIT_MULTIPLE * timeout`` seconds.
        """
        if self._claim_sentinel():
            return True

        own_pid = os.getpid()
        start = time.monotonic()
        next_log_at = start + timeout
        hard_deadline = start + timeout * _LIVE_HOLDER_WAIT_MULTIPLE
        while True:
            if self._try_load(checksum):
                return False

            holder_pid = self._read_lock_pid()
            if holder_pid is not None and holder_pid != own_pid and _pid_alive(holder_pid):
                now = time.monotonic()
                if now > hard_deadline:
                    raise TimeoutError(
                        f"Build lock {self._lock_path()} has been held by live pid "
                        f"{holder_pid} for over {timeout * _LIVE_HOLDER_WAIT_MULTIPLE:.1f}s "
                        f"({_LIVE_HOLDER_WAIT_MULTIPLE}x lock_timeout). Refusing to wait "
                        f"indefinitely. If that process is confirmed gone, remove "
                        f"{self._lock_path()} manually and retry."
                    )
                if now >= next_log_at:
                    _logger.warning(
                        "Build lock %s still held by live pid %d after %.1fs; waiting "
                        "rather than taking over.",
                        self._lock_path(),
                        holder_pid,
                        now - start,
                    )
                    next_log_at += timeout
                time.sleep(poll_interval)
                continue

            if not self._lock_is_stale(timeout):
                time.sleep(poll_interval)
                continue

            # holder_pid is dead, unreadable, or (pid recycling) our own, and
            # the sentinel is stale. TOCTOU guard: re-read immediately before
            # unlinking -- another waiter may have already reclaimed a fresh,
            # live sentinel in the interim; unlinking that would kill it.
            if self._read_lock_pid() != holder_pid:
                continue
            _logger.warning(
                "Build lock %s has no live holder and is stale; taking over.",
                self._lock_path(),
            )
            try:
                self._lock_path().unlink()
            except FileNotFoundError:
                pass
            if self._claim_sentinel():
                return True
            # Another process reclaimed it first -- reassess from the top.

    def _start_heartbeat(self, timeout: float) -> None:
        """Starts a daemon thread that periodically refreshes the sentinel's mtime.

        A waiter treats a holder's sentinel as possibly abandoned once its
        mtime goes stale (see :meth:`_lock_is_stale`); refreshing it while
        genuinely building corroborates :func:`_pid_alive`, covering that
        check's own false negatives (e.g. a pid query blocked or denied for
        a transient reason) with an independent liveness signal.

        Args:
            timeout: The caller's ``lock_timeout``; the refresh interval is
                a small fraction of it, bounded to a sane range.
        """
        interval = max(_HEARTBEAT_MIN_INTERVAL, min(timeout / 4, _HEARTBEAT_MAX_INTERVAL))
        lock_path = self._lock_path()
        stop = threading.Event()

        def _beat() -> None:
            while not stop.wait(interval):
                try:
                    os.utime(lock_path, None)
                except FileNotFoundError:
                    return  # sentinel is gone -- released or taken over, nothing left to refresh
                except OSError:
                    pass  # transient (e.g. a momentary sharing violation) -- keep trying

        self._heartbeat_stop = stop
        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stops the heartbeat thread started by :meth:`_start_heartbeat`, if any.

        Guards ``join()`` against a thread object that was constructed but
        never actually started -- if ``Thread.start()`` itself raised (e.g.
        the OS refused a new thread), ``_heartbeat_thread`` is still set, and
        an unguarded ``join()`` would raise "cannot join thread before it is
        started". That would mask the original ``start()`` error *and* skip
        :meth:`_release_lock` on the next line (this runs inside a
        ``finally``), leaving a live-pid sentinel that stalls every other
        waiter for up to ``_LIVE_HOLDER_WAIT_MULTIPLE * lock_timeout``.
        """
        if self._heartbeat_stop is None:
            return
        assert self._heartbeat_thread is not None  # set together in _start_heartbeat
        self._heartbeat_stop.set()
        try:
            self._heartbeat_thread.join(timeout=1.0)
        except RuntimeError:
            pass  # Thread.start() itself never succeeded -- nothing to join
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    def _release_lock(self) -> None:
        """Removes the sentinel this process created in :meth:`_acquire_lock`.

        A no-op unless this instance actually created (or took over) the
        sentinel: releasing one this process never owned would let a third
        process immediately win a fresh race into a directory whose real
        (still-live or just-finished) builder never asked to release it.
        The ownership flag alone is not trusted for the unlink itself: the
        sentinel's *content* is re-checked to still be this process's own
        pid first, since a takeover elsewhere (see :meth:`_acquire_lock`'s
        TOCTOU guard) could in principle have replaced it in between --
        unlinking unconditionally here would then delete a third party's
        live sentinel instead of this process's own.
        """
        if not self._owns_lock:
            return
        if self._read_lock_pid() == os.getpid():
            try:
                self._lock_path().unlink()
            except FileNotFoundError:
                pass
        self._owns_lock = False

    @staticmethod
    def _pid_from_sibling_name(name: str) -> int | None:
        """Extracts the embedded pid from a ``<...>.<kind>.<pid>.tmp`` sibling name."""
        stem = name.removesuffix(".tmp")
        try:
            return int(stem.rsplit(".", 1)[-1])
        except ValueError:
            return None

    def _sweep_stale_siblings(self) -> None:
        """Removes leftover build/trash temp siblings whose owning pid has died.

        Both :meth:`_build`'s ``.build.<pid>.tmp`` and :meth:`_publish`'s
        ``.trash.<pid>.tmp`` are full, ``map_size``-presized LMDB directories
        (several GB for a real dataset) — a crash before cleanup otherwise
        leaks that disk space forever. Only called while holding the
        exclusive build lock (from :meth:`_build`), so a sibling whose pid
        is still alive belongs to a build genuinely in progress elsewhere
        and must be left untouched.
        """
        prefix = self._lmdb_path.name
        for pattern in (f"{prefix}.build.*.tmp", f"{prefix}.trash.*.tmp"):
            for candidate in self._cache_dir.glob(pattern):
                pid = self._pid_from_sibling_name(candidate.name)
                if pid is None or pid == os.getpid() or _pid_alive(pid):
                    continue
                try:
                    shutil.rmtree(candidate)
                except OSError:
                    pass  # best-effort; retried by the next build that reaches here

    def _build(
        self,
        checksum: str,
        length: int,
        map_size: int,
        build_fn: Callable[[LMDBCacheBuildContext], None],
        use_tqdm: bool,
    ) -> None:
        """Builds into a temp sibling directory, then atomically publishes it.

        *build_fn* must populate it with exactly *length* entries.
        :attr:`_lmdb_path` itself is never opened for writing until the new
        database is fully populated and closed, so a build that fails here
        never touches (or half-deletes) whatever was already at that path —
        reaching this method at all already means :meth:`_acquire_lock`
        verified no other process can still be live-writing there.
        """
        self._sweep_stale_siblings()

        tmp_path = self._lmdb_path.with_name(f"{self._lmdb_path.name}.build.{os.getpid()}.tmp")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)  # leftover of an earlier crashed attempt, same pid: ours

        env = lmdb.open(str(tmp_path), map_size=map_size)
        try:
            ctx = LMDBCacheBuildContext(env=env, use_tqdm=use_tqdm)
            build_fn(ctx)

            # __checksum__ written last so an interrupted build reads as stale, not complete.
            txn = env.begin(write=True)
            txn.put(b"__length__", str(length).encode())
            for k, v in self._metadata.items():
                txn.put(f"__{k}__".encode(), str(v).encode())
            txn.put(b"__checksum__", checksum.encode())
            txn.commit()
        finally:
            env.close()

        self._publish(tmp_path)
        self._length = length

    def _publish(self, tmp_path: Path) -> None:
        """Atomically moves a finished build from *tmp_path* into :attr:`_lmdb_path`.

        Never ``rmtree``s :attr:`_lmdb_path` in place: a pre-existing
        directory there is moved *aside* to a pid-suffixed trash sibling via
        ``os.replace`` (a rename, not a delete -- safe even against a
        reader/mapper still holding files inside it open, unlike unlinking
        them directly), the finished build is renamed into the now-vacant
        real path, and only then is the trash sibling opportunistically
        removed. This leaves no window where anything is unlinked out from
        under an open handle, at the cost of transiently roughly doubling
        peak disk usage (old + new both present) until that last cleanup
        runs -- or indefinitely, if it can never win the race against a
        lingering reader; see :meth:`_sweep_stale_siblings` for the backstop.

        Reaching here means :meth:`_acquire_lock` gave this process the
        exclusive, live-verified build lock, so any pre-existing directory
        at :attr:`_lmdb_path` is provably not another process's in-progress
        build — only a genuinely stale (failed-checksum or corrupt)
        leftover can land here. If moving it aside or moving the new build
        into place fails anyway (e.g. a lingering open handle on Windows),
        that is raised as a clear error instead of silently discarding the
        build just finished.

        Args:
            tmp_path: The completed build's temporary directory.

        Raises:
            RuntimeError: If the stale target could not be moved aside, or
                the finished build could not be moved into place.
        """
        trash_path = None
        if self._lmdb_path.exists():
            trash_path = self._lmdb_path.with_name(
                f"{self._lmdb_path.name}.trash.{os.getpid()}.tmp"
            )
            try:
                os.replace(self._lmdb_path, trash_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot move the stale cache at {self._lmdb_path} aside to publish a "
                    "fresh build; leaving both in place. Investigate and remove it manually, "
                    "then retry."
                ) from exc
        try:
            os.replace(tmp_path, self._lmdb_path)
        except OSError as exc:
            raise RuntimeError(
                f"Built cache at {tmp_path} but could not move it to {self._lmdb_path}."
            ) from exc
        if trash_path is not None:
            try:
                shutil.rmtree(trash_path)
            except OSError:
                pass  # best-effort: a lingering reader may still hold it open
