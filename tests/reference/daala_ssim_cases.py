"""Synthetic plane pairs for daala-SSIM parity checks.

Shared by ``tests/test_ssim.py`` and ``gen_daala_ssim_expected.py`` so both
sides score byte-identical inputs. Sizes are chosen to exercise every branch of
daala's ``gaussian_filter_init``: the 9-tap kernel at H=256, the ``max_len``
cap, the degenerate zero-length kernel, and the height-only sigma (a portrait
and landscape pair of the same content must score differently). A further four
cases pair a reference with a structurally-correlated-but-imperfect distortion
(additive noise at two strengths, a 1px shift, a gamma curve) so the
cross-covariance term (``sigma_xy``) actually drives the score, rather than
the near-zero regime independent noise leaves it in.
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
    {
        "name": "noise_snr_high_256x256",
        "hw": (256, 256),
        "seed": 10,
        "mode": "additive_noise",
        "std": 40.0,
    },
    {
        "name": "noise_snr_low_256x256",
        "hw": (256, 256),
        "seed": 11,
        "mode": "additive_noise",
        "std": 120.0,
    },
    {"name": "subpixel_shift_192x256", "hw": (192, 256), "seed": 12, "mode": "shift"},
    {"name": "gamma_shift_224x160", "hw": (224, 160), "seed": 13, "mode": "gamma", "gamma": 3.5},
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
    if mode == "additive_noise":
        # Gaussian noise on a correlated reference -- drives sigma_xy well away
        # from ~0, unlike two independent uniform planes.
        noise = rng.normal(0.0, case["std"], size=(h, w))
        return a, np.clip(a.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    if mode == "shift":
        # Smooth first so a 1px shift keeps real local correlation -- shifting
        # raw iid noise would decorrelate completely and defeat the point.
        f = a.astype(np.float64)
        acc = np.zeros_like(f)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += np.roll(np.roll(f, dy, 0), dx, 1)
        smoothed = np.clip(acc / 9.0, 0, 255).astype(np.uint8)
        return smoothed, np.roll(smoothed, 1, axis=1)
    if mode == "gamma":
        b = 255.0 * (a.astype(np.float64) / 255.0) ** case["gamma"]
        return a, np.clip(b, 0, 255).round().astype(np.uint8)
    return a, rng.integers(0, 256, size=(h, w), dtype=np.uint8)
