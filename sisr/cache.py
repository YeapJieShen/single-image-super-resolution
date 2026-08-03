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


def _pid_alive(pid: int) -> bool:
    """Best-effort, portable check for whether *pid* names a running process.

    ``os.kill(pid, 0)`` is the standard POSIX liveness idiom, but on Windows
    CPython's ``os.kill`` maps arbitrary signal numbers (including 0) onto
    ``TerminateProcess`` -- calling it here could actually kill a live
    builder rather than merely check it. So on that platform this queries
    the process table via ``ctypes``/``OpenProcess`` instead.

    Args:
        pid: Process id to check.

    Returns:
        ``True`` if a process with this pid is currently running.
    """
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
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

            def _submit():
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
    alive (its pid running, corroborated by a heartbeat that refreshes the
    sentinel's mtime during the build) — a losing process instead keeps
    waiting past *lock_timeout*, logging once, rather than racing a slow but
    healthy build. Only a holder whose pid has died *and* whose sentinel has
    gone stale for longer than *lock_timeout* is treated as abandoned and
    taken over. Waiting forever on a genuinely stale lock is the one failure
    mode this must never have; rebuilding a cache that already exists is
    merely wasted time.

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

        if not self._try_load(checksum):
            if build_fn is None:
                raise RuntimeError(
                    f"No valid LMDB cache found at {self._lmdb_path} and no build_fn was provided."
                )
            if self._acquire_lock(checksum, lock_poll_interval, lock_timeout):
                self._start_heartbeat(lock_timeout)
                try:
                    # Re-check: a concurrent builder may have finished (and released
                    # its lock) between our first _try_load and winning this one.
                    if not self._try_load(checksum):
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

    def _try_load(self, checksum: str) -> bool:
        """Validates an existing LMDB against *checksum*; ``True`` if ready to use.

        Only a corruption-specific failure to *open* the environment (bad
        file header, wrong on-disk version — see :data:`_CORRUPTION_ERRORS`)
        is presumed genuine corruption, dropping the directory so a later
        call can rebuild it. Every other failure, opening or reading, is
        treated as merely not-ready-yet without deleting anything: opened
        with ``lock=False``, this can raise a bare ``lmdb.Error`` ("already
        open in this process") when this same process holds another handle
        on the same path, or fail on Windows while a concurrent writer *in
        another process* is still active (the exact scenario
        :meth:`_acquire_lock` exists for) — in neither case is anything
        actually corrupt. Deleting the directory there would destroy that
        other process's in-progress or just-finished build instead of merely
        costing this call a retry. Even the corruption-cleanup rmtree itself
        is guarded: a failure there (e.g. a lingering open handle) must not
        escape as an unhandled exception either.
        """
        if not self._lmdb_path.exists():
            return False
        try:
            env = lmdb.open(str(self._lmdb_path), readonly=True, lock=False)
        except _CORRUPTION_ERRORS:
            try:
                if self._lmdb_path.exists():
                    shutil.rmtree(self._lmdb_path)
            except OSError:
                pass
            return False
        except (lmdb.Error, OSError):
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
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
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

        A live holder — its pid still running — is never preempted: this
        waits past *timeout*, logging once, rather than racing a slow but
        healthy build. Takeover only happens once the holder's pid is
        confirmed dead *and* its sentinel's mtime (refreshed by the
        holder's heartbeat while it builds, see :meth:`_start_heartbeat`)
        has gone stale for longer than *timeout* — i.e. it crashed without
        releasing the lock.

        Returns:
            ``True`` if the caller should proceed to build — either it won
            the lock outright, or it took over one abandoned by a dead
            holder. ``False`` if a concurrent builder produced a valid cache
            while waiting — :meth:`_try_load` has already refreshed
            :attr:`_length` in that case.
        """
        if self._claim_sentinel():
            return True

        warned_waiting = warned_stale = False
        deadline = time.monotonic() + timeout
        while True:
            if self._try_load(checksum):
                return False

            holder_pid = self._read_lock_pid()
            if holder_pid is not None and _pid_alive(holder_pid):
                if not warned_waiting and time.monotonic() > deadline:
                    _logger.warning(
                        "Build lock %s held by live pid %d past the %.1fs timeout; "
                        "waiting rather than taking over.",
                        self._lock_path(),
                        holder_pid,
                        timeout,
                    )
                    warned_waiting = True
                time.sleep(poll_interval)
                continue

            if not self._lock_is_stale(timeout):
                time.sleep(poll_interval)
                continue

            if not warned_stale:
                _logger.warning(
                    "Build lock %s has no live holder and is stale; taking over.",
                    self._lock_path(),
                )
                warned_stale = True
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
        genuinely building proves this process is still alive even in the
        pid-alive check's absence (e.g. a very short-lived poll window).

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
                except OSError:
                    return

        self._heartbeat_stop = stop
        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stops the heartbeat thread started by :meth:`_start_heartbeat`, if any."""
        if self._heartbeat_stop is None:
            return
        self._heartbeat_stop.set()
        self._heartbeat_thread.join(timeout=1.0)
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    def _release_lock(self) -> None:
        """Removes the sentinel this process created in :meth:`_acquire_lock`.

        A no-op unless this instance actually created (or took over) the
        sentinel: releasing one this process never owned would let a third
        process immediately win a fresh race into a directory whose real
        (still-live or just-finished) builder never asked to release it.
        """
        if not self._owns_lock:
            return
        try:
            self._lock_path().unlink()
        except FileNotFoundError:
            pass
        self._owns_lock = False

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

        Reaching here means :meth:`_acquire_lock` gave this process the
        exclusive, live-verified build lock, so any pre-existing directory
        at :attr:`_lmdb_path` is provably not another process's in-progress
        build — only a genuinely stale (failed-checksum or corrupt)
        leftover can land here. If clearing or replacing it fails anyway
        (e.g. a lingering open handle on Windows), that is raised as a
        clear error instead of silently discarding the build just finished.

        Args:
            tmp_path: The completed build's temporary directory.

        Raises:
            RuntimeError: If the stale target could not be cleared, or the
                finished build could not be moved into place.
        """
        if self._lmdb_path.exists():
            try:
                shutil.rmtree(self._lmdb_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot replace stale cache at {self._lmdb_path}: still in use "
                    "elsewhere. Not deleting it -- remove it manually once nothing "
                    "holds it open, then retry."
                ) from exc
        try:
            os.replace(tmp_path, self._lmdb_path)
        except OSError as exc:
            raise RuntimeError(
                f"Built cache at {tmp_path} but could not move it to {self._lmdb_path}."
            ) from exc
