"""Synthetic plane pairs for daala-SSIM parity checks.

Shared by ``tests/test_ssim.py`` and ``gen_daala_ssim_expected.py`` so both
sides score byte-identical inputs. Sizes are chosen to exercise every branch of
daala's ``gaussian_filter_init``: the 9-tap kernel at H=256, the ``max_len``
cap, the degenerate zero-length kernel, and the height-only sigma (a portrait
and landscape pair of the same content must score differently).
"""

import numpy as np

#: name, (height, width), seed, and how the second plane differs from the first.
CASES: list[dict] = [
    {"name": "noise_256x256", "hw": (256, 256), "seed": 1, "mode": "noise"},
    {"name": "noise_512x512", "hw": (512, 512), "seed": 2, "mode": "noise"},
    {"name": "landscape_200x320", "hw": (200, 320), "seed": 3, "mode": "noise"},
    {"name": "portrait_320x200", "hw": (320, 200), "seed": 3, "mode": "noise"},
    {"name": "odd_101x67", "hw": (101, 67), "seed": 4, "mode": "noise"},
    {"name": "narrow_tall_512x4", "hw": (512, 4), "seed": 5, "mode": "noise"},
    {"name": "tiny_8x8", "hw": (8, 8), "seed": 6, "mode": "noise"},
    {"name": "identical_128x128", "hw": (128, 128), "seed": 7, "mode": "identical"},
    {"name": "constant_offset_64x64", "hw": (64, 64), "seed": 8, "mode": "constant"},
    {"name": "blurred_320x480", "hw": (320, 480), "seed": 9, "mode": "blur"},
]


def make_planes(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(reference, distorted)`` uint8 plane pair for *case*.

    Args:
        case: One entry of :data:`CASES`.

    Returns:
        Two ``(H, W)`` ``uint8`` arrays. Deterministic for a given case —
        ``numpy.random.Generator(PCG64)`` is reproducible across platforms.
    """
    h, w = case["hw"]
    rng = np.random.default_rng(case["seed"])
    mode = case["mode"]

    if mode == "constant":
        return np.full((h, w), 128, np.uint8), np.full((h, w), 130, np.uint8)

    a = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    if mode == "identical":
        return a, a.copy()
    if mode == "blur":
        # Cheap 3x3 box blur, so the pair differs structurally rather than by
        # independent noise -- an SSIM well inside (0, 1).
        f = a.astype(np.float64)
        acc = np.zeros_like(f)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += np.roll(np.roll(f, dy, 0), dx, 1)
        return a, np.clip(acc / 9.0, 0, 255).astype(np.uint8)
    return a, rng.integers(0, 256, size=(h, w), dtype=np.uint8)
