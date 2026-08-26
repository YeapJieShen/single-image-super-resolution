"""Synthetic and real-image plane pairs for daala-SSIM parity checks.

Shared by ``tests/metrics/test_ssim.py`` and ``gen_daala_ssim_expected.py`` so both
sides score byte-identical inputs.

:data:`CASES`/:func:`make_planes` are synthetic. Sizes are chosen to exercise
every branch of daala's ``gaussian_filter_init``: the 9-tap kernel at H=256,
the ``max_len`` cap, the degenerate zero-length kernel, and the height-only
sigma (a portrait and landscape pair of the same content must score
differently). A further four cases pair a reference with a
structurally-correlated-but-imperfect distortion (additive noise at two
strengths, a 1px shift, a gamma curve) so the cross-covariance term
(``sigma_xy``) actually drives the score, rather than the near-zero regime
independent noise leaves it in.

:data:`REAL_SETS`/:func:`discover_real_cases`/:func:`make_real_planes` are
real-image cases built from Set5/Set14/BSD100, reproducing the SRResNet
scoring pipeline's **bicubic baseline** (no model needed, so these stay
regenerable without a checkpoint) exactly: mod-crop to a multiple of 4, a
down-then-up bicubic round trip, a 4px border crop, studio-range Y, quantized
to uint8.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sisr.colorspace import rgb_to_ycbcr_studio
from sisr.metrics.ssim import quantize_u8
from sisr.utils.imresize import resize

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


#: Per-set HR directories real cases are built from, repo-root-relative —
#: the same sets and scale (4) the SRResNet Ledig-comparison arc scores.
REAL_SETS: dict[str, str] = {
    "Set5": "data/Set5_HR",
    "Set14": "data/Set14_HR",
    "BSD100": "data/BSD100_HR",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCALE = 4  # SRResNet's scale -- both the mod-crop modulus and the bicubic round trip.
_CROP_BORDER = 4  # Ledig et al.'s border strip, matching SRResNetEvalConfig.crop_border.


def discover_real_cases(repo_root: Path = _REPO_ROOT) -> list[dict]:
    """List one case per image across :data:`REAL_SETS`, sorted for determinism.

    Args:
        repo_root: Repository root the :data:`REAL_SETS` paths are relative to.

    Returns:
        Dicts with ``name`` (``"<Set>/<stem>"``, the key into
        ``daala_ssim_expected.json``), ``set``, and ``path`` (the HR image
        file). ``[]`` if none of the set directories exist under
        *repo_root* -- ``tests/metrics/test_ssim.py`` treats that as "data/ absent"
        (skip cleanly), distinct from a set directory existing but holding
        no images (an error).
    """
    cases = []
    for set_name, rel_dir in REAL_SETS.items():
        img_dir = repo_root / rel_dir
        if not img_dir.is_dir():
            continue
        for path in sorted(img_dir.glob("*.png")):
            cases.append({"name": f"{set_name}/{path.stem}", "set": set_name, "path": path})
    return cases


def _quantized_studio_y(rgb_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 HWC -> studio-range Y, quantized to uint8 -- the daala input.

    The same conversion :class:`~sisr.training.lightning_module.SRLightning`
    applies before scoring: normalise to ``[0, 1]``, ``rgb_to_ycbcr_studio``,
    ``quantize_u8``.
    """
    t = torch.from_numpy(rgb_u8).permute(2, 0, 1).float().div(255.0)[None]
    y = rgb_to_ycbcr_studio(t)[:, 0:1]
    return quantize_u8(y)[0, 0].numpy().astype(np.uint8)


def make_real_planes(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(bicubic, hr)`` uint8 Y-plane pair for a real-image *case*.

    Reproduces the SRResNet validation scoring pipeline exactly, substituting
    the bicubic baseline for model output (see module docstring):

    1. Mod-crop HR to a multiple of ``scale`` (:class:`sisr.datasets.srresnet.
       ValidationDataset`'s convention).
    2. :func:`sisr.utils.imresize.resize` down by ``scale``, then back up by
       ``scale`` -- the model-free reconstruction.
    3. Crop ``crop_border`` px from each edge of both.
    4. Studio-range Y (:func:`sisr.colorspace.rgb_to_ycbcr_studio`), quantized
       to uint8 (:func:`sisr.metrics.ssim.quantize_u8`) -- the bytes the daala C scores.

    Args:
        case: One entry of :func:`discover_real_cases`.

    Returns:
        Two ``(H, W)`` ``uint8`` arrays: the bicubic reconstruction and the
        HR reference.
    """
    arr = np.array(Image.open(case["path"]).convert("RGB"))
    h, w = arr.shape[:2]
    hr = arr[: h - h % _SCALE, : w - w % _SCALE, :]
    hr_h, hr_w = hr.shape[:2]
    lr = resize(hr, (hr_h // _SCALE, hr_w // _SCALE))
    bicubic = resize(lr, (hr_h, hr_w))

    n = _CROP_BORDER
    hr_c, bicubic_c = hr[n:-n, n:-n, :], bicubic[n:-n, n:-n, :]
    return _quantized_studio_y(bicubic_c), _quantized_studio_y(hr_c)


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
