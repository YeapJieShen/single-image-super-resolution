"""Colorspace conversions and the LMDB cache infrastructure.

BT.601 full-range YCbCr is the project's working chroma space. Pure
conversion functions live here so the :class:`~sisr.processors.SRProcessor`
subclasses and any external callers can use them without depending on
Lightning or model code.

:class:`LMDBCache` is a checksum-validated key-value store used by
:class:`~sisr.datasets.srcnn.TrainDataset` to persist precomputed
LR/HR sub-image pairs.
"""

import os
import shutil
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import lmdb
import torch
from tqdm.auto import tqdm

# BT.601 full-range RGB <-> YCbCr conversion (ITU-R Rec. BT.601-7).
# Both colorspaces are normalised to [0, 1]. Cb and Cr have a +0.5 offset on
# their stored representation so that signed chroma values in [-0.5, +0.5]
# map onto unsigned [0, 1]. This is the *full-range* (a.k.a. "JPEG") variant
# of BT.601 — distinct from the studio-range variant which scales Y to
# [16/255, 235/255] and Cb/Cr to [16/255, 240/255].

_RGB_TO_Y = (0.299, 0.587, 0.114)
_RGB_TO_CB = (-0.169, -0.331, 0.500)  # output offset +0.5
_RGB_TO_CR = (0.500, -0.419, -0.081)  # output offset +0.5
_YCBCR_TO_R = 1.402  # cr coefficient
_YCBCR_TO_G_CB = -0.344  # cb coefficient
_YCBCR_TO_G_CR = -0.714  # cr coefficient
_YCBCR_TO_B = 1.772  # cb coefficient


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """Convert a normalised RGB tensor to YCbCr (BT.601 full-range).

    Args:
        img (torch.Tensor): RGB tensor of shape ``(B, 3, H, W)`` with
            values in ``[0, 1]``.

    Returns:
        torch.Tensor: YCbCr tensor of shape ``(B, 3, H, W)`` in
        ``[0, 1]`` with Cb/Cr offset by +0.5.
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = _RGB_TO_Y[0] * r + _RGB_TO_Y[1] * g + _RGB_TO_Y[2] * b
    cb = _RGB_TO_CB[0] * r + _RGB_TO_CB[1] * g + _RGB_TO_CB[2] * b + 0.5
    cr = _RGB_TO_CR[0] * r + _RGB_TO_CR[1] * g + _RGB_TO_CR[2] * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert a YCbCr tensor to RGB (BT.601 full-range, clamped to [0, 1]).

    Args:
        img (torch.Tensor): YCbCr tensor of shape ``(B, 3, H, W)`` with
            Cb/Cr offset by +0.5.

    Returns:
        torch.Tensor: RGB tensor of shape ``(B, 3, H, W)`` clamped to
        ``[0, 1]``.
    """
    y, cb, cr = img[:, 0:1], img[:, 1:2] - 0.5, img[:, 2:3] - 0.5
    r = y + _YCBCR_TO_R * cr
    g = y + _YCBCR_TO_G_CB * cb + _YCBCR_TO_G_CR * cr
    b = y + _YCBCR_TO_B * cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


class LMDBCacheBuildContext:
    """Context object passed to the ``build_fn`` callback of :class:`LMDBCache`.

    Provides helpers for writing data into the LMDB being built.

    Args:
        env (lmdb.Environment): The open LMDB environment (writable).
        use_tqdm (bool): Whether progress bars should be displayed.
    """

    def __init__(self, env: lmdb.Environment, use_tqdm: bool = False):
        self.env = env
        self.use_tqdm = use_tqdm

    def write_batch(self, pairs: Sequence[tuple[str, bytes]]) -> None:
        """Writes a batch of key-value pairs in a single transaction.

        Args:
            pairs (Sequence[tuple[str, bytes]]): Sequence of ``(key, value)`` tuples to write.
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
    ):
        """Processes *items* and writes the results to LMDB.

        The worker function *process_fn* must be a top-level (picklable)
        callable returning a list of ``(key, value_bytes)`` tuples.  With more
        than one effective worker a ``ProcessPoolExecutor`` runs a sliding
        window of *num_workers* in-flight jobs while the main process writes
        completed results; with ``<= 1`` effective worker the jobs run inline in
        the calling process (no subprocess).

        Args:
            items (Sequence[Any]): One item per job (e.g. a list of image paths).
            process_fn (Callable[..., list[tuple[str, bytes]]]): Top-level callable invoked per
                item.  Receives ``(item, *extra_args)`` and returns keyed pairs.
            process_args (Sequence[Sequence[Any]], optional): Per-item extra arguments for
                *process_fn*. If ``None``, each job calls ``process_fn(item)``.
                If provided, must have the same length as *items* and
                each element is unpacked as positional args.
            num_workers (int | None): Maximum number of parallel worker
                processes.  ``None`` resolves to ``min(os.cpu_count() or 1,
                len(items))``.  When the effective count is ``<= 1`` (a single
                core, a lone item, or an explicit ``1``) the build runs inline
                with no ``ProcessPoolExecutor``.
            desc (str): Description shown in the ``tqdm`` progress bar.
        """
        n_items = len(items)
        if num_workers is None:
            num_workers = min(os.cpu_count() or 1, n_items)
        num_workers = min(num_workers, n_items)

        pbar = tqdm(total=n_items, desc=desc, unit="item") if self.use_tqdm else None

        # Inline (no-pool) build. With <= 1 effective worker the
        # ProcessPoolExecutor spawn/import cost — plus the hazard of nesting a
        # pool inside a test/xdist worker — isn't worth it, so run jobs here.
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
    already exists (matched by *checksum*).  If not, it calls
    ``build_fn`` to populate the database from scratch using a pool of
    worker processes.

    After construction the cache is read-only.  Call :meth:`get` to
    retrieve individual entries, or :meth:`get_env` for direct LMDB
    access.

    Args:
        cache_dir (str | Path): Parent directory for the LMDB database.
        name (str): Prefix used in the LMDB folder name
            (e.g. ``'srcnn_patches'``).
        checksum (str): Hex digest that uniquely identifies the current
            configuration.  A mismatch triggers a rebuild.
        length (int): Total number of entries that will be stored.
        map_size (int): Maximum size of the LMDB database in bytes.
        metadata (dict[str, str] | None): Extra key-value pairs to persist alongside the data
            (e.g. ``{'channels': '3', 'subimg_size': '33'}``).
        build_fn (Callable[[LMDBCacheBuildContext], None] | None): A callable that populates the
            database.  It receives a single :class:`LMDBCacheBuildContext` argument exposing
            a ``write_batch`` helper and an ``env`` handle.  If
            ``None`` and no valid cache is found, a ``RuntimeError``
            is raised.
        use_tqdm (bool): Whether to display a progress bar during the build.
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
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lmdb_path = self._cache_dir / f"{name}_{checksum[:16]}"
        self._length = length
        self._metadata = metadata or {}
        self._env: lmdb.Environment | None = None

        if not self._try_load(checksum):
            if build_fn is None:
                raise RuntimeError(
                    f"No valid LMDB cache found at {self._lmdb_path} and no build_fn was provided."
                )
            self._build(checksum, length, map_size, build_fn, use_tqdm)

    @property
    def path(self) -> Path:
        """Path to the LMDB database directory.

        Returns:
            The path where the LMDB database is stored.
        """
        return self._lmdb_path

    @property
    def length(self) -> int:
        """Number of entries stored in the cache.

        Returns:
            The number of entries.
        """
        return self._length

    def get_env(self) -> lmdb.Environment:
        """Returns the LMDB environment, opening it lazily on first call.

        Each ``DataLoader`` worker process must call this independently
        because LMDB memory-mapped environments cannot be shared across
        processes created via ``spawn``.

        Returns:
            A read-only LMDB environment.
        """
        if self._env is None:
            self._env = lmdb.open(str(self._lmdb_path), readonly=True, lock=False)
        return self._env

    def get(self, key: str) -> bytes | None:
        """Reads a single value from the cache.

        Args:
            key (str): The string key to look up.

        Returns:
            The raw bytes stored under *key*, or ``None`` if absent.
        """
        env = self.get_env()
        with env.begin(write=False, buffers=True) as txn:
            buf = txn.get(key.encode())
            if buf is None:
                return None
            return bytes(buf)

    def get_batch(self, keys: Sequence[str]) -> list[bytes | None]:
        """Reads multiple values from the cache in a single transaction.

        Args:
            keys (Sequence[str]): Sequence of string keys to look up.

        Returns:
            A list of raw bytes (or ``None`` for missing keys), in the
            same order as *keys*.
        """
        env = self.get_env()
        results = []
        with env.begin(write=False, buffers=True) as txn:
            for key in keys:
                buf = txn.get(key.encode())
                results.append(bytes(buf) if buf is not None else None)
        return results

    def _try_load(self, checksum: str) -> bool:
        """Validates an existing LMDB by comparing its stored checksum.

        Args:
            checksum (str): The expected checksum to validate against.

        Returns:
            ``True`` if the cache is valid and ready to use.
        """
        if not self._lmdb_path.exists():
            return False
        try:
            env = lmdb.open(str(self._lmdb_path), readonly=True, lock=False)
            with env.begin(write=False) as txn:
                stored = txn.get(b"__checksum__")
                if stored is None or stored.decode() != checksum:
                    env.close()
                    return False
                self._length = int(txn.get(b"__length__").decode())
            env.close()
            return True
        except (lmdb.Error, OSError):
            # Only a genuinely unreadable cache (LMDB corruption, disk/OS
            # error) counts as "stale" — drop it and rebuild. Anything else
            # (KeyboardInterrupt, data-shape bugs, ...) must propagate so a
            # broken system isn't silently mistaken for a stale cache.
            if self._lmdb_path.exists():
                shutil.rmtree(self._lmdb_path)
            return False

    def _build(
        self,
        checksum: str,
        length: int,
        map_size: int,
        build_fn: Callable[[LMDBCacheBuildContext], None],
        use_tqdm: bool,
    ):
        """Creates a fresh LMDB and delegates population to *build_fn*.

        Args:
            checksum (str): The checksum to store for future validation.
            length (int): The number of entries that will be stored.
            map_size (int): The maximum size of the LMDB database in bytes.
            build_fn (Callable[[LMDBCacheBuildContext], None]): A callable that populates the
                database.  It receives a single :class:`LMDBCacheBuildContext` argument exposing
                a ``write_batch`` helper and an ``env`` handle. Must populate the database with
                exactly *length* entries.
            use_tqdm (bool): Whether to display a progress bar during the build.
        """
        if self._lmdb_path.exists():
            shutil.rmtree(self._lmdb_path)

        env = lmdb.open(str(self._lmdb_path), map_size=map_size)

        ctx = LMDBCacheBuildContext(env=env, use_tqdm=use_tqdm)
        build_fn(ctx)

        # Write metadata — __checksum__ last for incomplete-build detection
        txn = env.begin(write=True)
        txn.put(b"__length__", str(length).encode())
        for k, v in self._metadata.items():
            txn.put(f"__{k}__".encode(), str(v).encode())
        txn.put(b"__checksum__", checksum.encode())
        txn.commit()
        env.close()

        self._length = length
