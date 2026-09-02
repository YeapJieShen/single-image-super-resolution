"""Whole-image derived planes, cached as a sibling of the raw-HR cache.

Both train datasets need a plane derived from the **whole** HR image: SRResNet
the bicubic downscale, SRCNN that downscale bicubically restored to HR size.
Degrading an already-extracted crop instead resolves the kernel's edge taps
against the crop border rather than the real neighbouring pixels, which is not
what the surveyed reference pipelines do and not what produced the numbers this
project compares against.

**A sibling cache, never a second value inside the raw-HR one.**
:func:`~sisr.datasets.hr_cache.compute_checksum` states that no degradation
parameter enters its hash, and every shipped config points every architecture at
one shared cache directory. A scale-dependent plane living in there would be
invalidated by nothing -- build at x4, run at x3, same checksum, wrong pixels.
So the derived plane gets its own database, whose *name* and *checksum* both
carry kind, scale and :data:`IMRESIZE_VERSION`. A stale one is unreachable
rather than silently wrong.

**Sizing matters more than it looks.** A shuffling training loader touches the
whole cache, so what governs throughput is whether the raw-HR cache *plus* this
plane fits in page cache. ``'lr'`` costs ``1/scale**2`` of the HR pixels and is
negligible; ``'bicubic'`` is HR-sized and roughly doubles the working set.
Crossing available RAM is a step change, not a gradient — measured at a **21x**
throughput loss on the far side, with the GPU idle and the loop I/O-bound. Size
the box against the dataset before choosing ``'bicubic'``; see
:mod:`sisr.datasets.srcnn`.

Deliberately torch-free, for the same reason :mod:`~sisr.datasets.hr_cache` is:
:func:`process_derived_image` is the function pickled to ``ProcessPoolExecutor``
build workers, and a spawned worker re-imports *its own defining module*.
"""

import hashlib
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from ..utils.imresize import resize

IMRESIZE_VERSION = 1
"""Bumped **by hand** whenever :mod:`sisr.utils.imresize` changes its output.

Deliberately not a hash of that module's source: a docstring or comment edit
would then invalidate every derived plane and trigger a multi-gigabyte rebuild.
That is the same failure mode that made the raw-HR cache key on name and size
only, and the fix's failure mode being common where the defect's is rare is
exactly why it was rejected there.
"""

KINDS = ("lr", "bicubic")
"""``'lr'`` -- the bicubic downscale by ``scale`` (SRResNet's model input).
``'bicubic'`` -- that downscale bicubically restored to HR size (SRCNN's)."""

HEADER = struct.Struct("<II")


def modcrop_extent(length: int, scale: int) -> int:
    """Returns the largest multiple of *scale* not exceeding *length*.

    The single owner of the authors' ``modcrop`` convention, in the one form
    its callers need: a cropped extent as a number, to slice by or to grid over.
    """
    return length - (length % scale)


def modcrop(arr: np.ndarray, scale: int) -> np.ndarray:
    """Crops *arr* so both spatial axes are whole multiples of *scale*.

    The authors' released ``demo_SR.m`` applies this to the ground truth
    **before** the bicubic round trip::

        im_gnd = modcrop(im, up_scale);
        im_l   = imresize(im_gnd, 1/up_scale, 'bicubic');
        im_b   = imresize(im_l,   up_scale,   'bicubic');

    Without it, an image whose height is not a multiple of *scale* is
    downsampled to ``h // scale`` rows and stretched back over ``h``, so the LR
    sampling grid does not align with the HR one.
    """
    h, w = arr.shape[:2]
    return arr[: modcrop_extent(h, scale), : modcrop_extent(w, scale)]


def derive(arr: np.ndarray, kind: str, scale: int) -> np.ndarray:
    """Produces one derived plane from a **whole** HR array.

    Args:
        arr: ``(H, W, 3)`` uint8 RGB, not yet modcropped.
        kind: One of :data:`KINDS`.
        scale: Upscaling factor.

    Returns:
        ``(H', W', 3)`` uint8 — ``(H, W)`` modcropped and divided by *scale*
        for ``'lr'``, or modcropped and round-tripped back to that size for
        ``'bicubic'``.

    Raises:
        ValueError: If *kind* is not in :data:`KINDS`.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}; got {kind!r}.")
    gnd = modcrop(arr, scale)
    h, w = gnd.shape[:2]
    lr = resize(np.ascontiguousarray(gnd), (h // scale, w // scale))
    if kind == "lr":
        return lr
    return resize(np.ascontiguousarray(lr), (h, w))


def derived_shape(height: int, width: int, kind: str, scale: int) -> tuple[int, int]:
    """The ``(h, w)`` :func:`derive` will produce, without deriving it."""
    h, w = modcrop_extent(height, scale), modcrop_extent(width, scale)
    return (h, w) if kind == "bicubic" else (h // scale, w // scale)


def cache_name(kind: str, scale: int) -> str:
    """The LMDB database name for this plane, sibling to the raw-HR one.

    Carries *kind*, *scale* and :data:`IMRESIZE_VERSION` so two derivations of
    the same images never share a database.
    """
    return f"derived_{kind}_x{scale}_v{IMRESIZE_VERSION}"


def process_derived_image(path: Path, idx: int, kind: str, scale: int) -> list[tuple[str, bytes]]:
    """Decodes one HR image, derives its plane, and returns its LMDB entry.

    Top-level (not a method) so it can be pickled by ``ProcessPoolExecutor`` on
    spawn platforms.

    Returns:
        ``[(f'{kind}_{idx:08d}', header + raw_bytes)]``, *header* being
        :data:`HEADER`-packed ``(h, w)`` of the derived plane.
    """
    arr = np.array(Image.open(path).convert("RGB"))
    out = derive(arr, kind, scale)
    h, w = out.shape[:2]
    return [(f"{kind}_{idx:08d}", HEADER.pack(h, w) + out.tobytes())]


def compute_checksum(img_paths: Sequence[Path], kind: str, scale: int) -> str:
    """SHA-256 over the file manifest **plus** kind, scale and imresize version.

    Unlike the raw-HR checksum, the derivation parameters are part of this hash:
    that is the whole reason this cache is separate. Keys on name and size only,
    for the reason stated in
    :func:`sisr.datasets.hr_cache.compute_checksum`.
    """
    manifest = ",".join(f"{p.name}:{p.stat().st_size}" for p in img_paths)
    canonical = "|".join(
        [manifest, f"kind={kind}", f"scale={scale}", f"imresize={IMRESIZE_VERSION}"]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def estimate_map_size(sizes: Sequence[tuple[int, int]], kind: str, scale: int) -> int:
    """LMDB ``map_size`` for the derived planes of images with these sizes.

    Exact derived sizes plus 10% slack, floored at 64 MiB — the contract
    :func:`sisr.datasets.hr_cache.estimate_map_size` documents.
    """
    total = 0
    for h, w in sizes:
        dh, dw = derived_shape(h, w, kind, scale)
        total += HEADER.size + dh * dw * 3
    return max(int(total * 1.1), 64 * 1024 * 1024)
