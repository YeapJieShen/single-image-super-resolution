"""Architecture-neutral raw-HR cache primitives, shared by every train dataset.

SRCNN's and SRResNet's train datasets cache the exact same thing for a given
directory of HR images: one whole decoded RGB array per image, headered with
its own ``(height, width)`` so the value is self-describing — recoverable
from the cache alone even if the source files are gone (this is what let a
``DIV2K_train_HR`` wipe be recovered from the cache once already). Unifying
the cache name, format tag, and build function here — rather than
duplicating them per architecture, or having one borrow the other's — means
the same image directory produces exactly one cache regardless of which
architecture's dataset builds it first.

Deliberately torch-free, like :mod:`sisr.cache`: :func:`process_hr_image` is
the function pickled to ``ProcessPoolExecutor`` build workers, and a spawned
worker re-imports *its own defining module* to unpickle it — this one, not
whichever dataset module happens to call it.
"""

import hashlib
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

CACHE_NAME = "hr_raw"
FORMAT_TAG = "hr_rgb_v1"

# (height, width), little-endian uint32 each, prefixed to every cached value so
# a per-image shape travels with its pixel data in one LMDB read. SRCNN's
# TrainDataset re-derives sizes from the source files before ever reading the
# cache and so never needs this back — it is kept uniformly anyway, because it
# is what makes a cached value self-describing rather than dependent on the
# source files still existing.
HEADER = struct.Struct("<II")


def process_hr_image(path: Path, idx: int) -> list[tuple[str, bytes]]:
    """Decodes one HR image and returns its single, headered LMDB entry.

    Top-level (not a method) so it can be pickled by ``ProcessPoolExecutor``
    on spawn platforms. Shared by every train dataset's build path, so a
    cache built by one architecture is reused, not rebuilt, by another over
    the same image files.

    Returns:
        A single-element list ``[(f'hr_{idx:08d}', header + raw_bytes)]``
        where *header* is :data:`HEADER`-packed ``(h, w)`` and *raw_bytes* is
        the ``(H, W, 3)`` uint8 RGB array's bytes.
    """
    arr = np.array(Image.open(path).convert("RGB"))
    h, w = arr.shape[:2]
    return [(f"hr_{idx:08d}", HEADER.pack(h, w) + arr.tobytes())]


def compute_checksum(img_paths: Sequence[Path]) -> str:
    """Computes a SHA-256 checksum over the file manifest plus :data:`FORMAT_TAG`.

    Shared by every train dataset so an identical file set hashes identically
    regardless of which architecture asks first. No degradation/grid
    parameter enters this hash — the cache stores whole raw images, unaffected
    by anything derived from them at read time.

    Returns:
        A hex-encoded SHA-256 digest string.
    """
    file_manifest = ",".join(f"{p.name}:{p.stat().st_size}" for p in img_paths)
    canonical = "|".join([file_manifest, f"format={FORMAT_TAG}"])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def estimate_map_size(sizes: Sequence[tuple[int, int]]) -> int:
    """Converts each cached image's ``(h, w)`` into the LMDB ``map_size`` to request.

    Exact sizes plus 10% slack. If it is ever exceeded, LMDB raises
    ``MapFullError`` mid-build and the checksum is never written, so the
    partial database reads as stale and is rebuilt rather than silently
    serving truncated data.

    Args:
        sizes: Each cached image's ``(height, width)``.

    Returns:
        The requested ``map_size`` in bytes, floored at 64 MiB.
    """
    total = sum(HEADER.size + h * w * 3 for h, w in sizes)
    return max(int(total * 1.1), 64 * 1024 * 1024)
