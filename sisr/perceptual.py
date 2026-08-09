"""LPIPS / DISTS perceptual metrics — the two conventions, in one place.

Both are *scoring* metrics, never losses: LPIPS postdates Ledig et al. by a
year, so an LPIPS-supervised generator is a different method, not SRGAN.

They exist here because an adversarial objective makes PSNR and SSIM worse by
design, so without a perceptual metric an SRGAN run has no quality signal at
all. Neither is a substitute for the paper's MOS study; they correlate with it
better than PSNR does, and that is the whole claim.

DISTS costs nothing — torchmetrics implements it over torchvision's VGG16,
both already required. LPIPS needs the ``lpips`` package (torchmetrics gates it
on ``_LPIPS_AVAILABLE``), hence the ``[perceptual]`` extra.
"""

from collections.abc import Callable

import torch

#: Metric name -> is-lower-better. Consumed by the checkpoint-monitor direction
#: check: PSNR/SSIM are higher-better and SRCheckpoint defaults to mode='max',
#: so monitoring one of these at the default would keep the *worst* model.
PERCEPTUAL_METRICS: dict[str, bool] = {"lpips": True, "dists": True}


def _lpips_fn() -> Callable:
    """Import LPIPS on demand, turning a missing extra into an actionable error.

    Indirected through a function (rather than a module-level import) so the
    import cost is only paid by runs that ask for it — this module is reachable
    from spawned DataLoader workers, which pay every import twice on Windows.
    """
    from torchmetrics.utilities.imports import _LPIPS_AVAILABLE

    if not _LPIPS_AVAILABLE:
        raise ImportError(
            "perceptual_metrics includes 'lpips', but torchmetrics reports the "
            "`lpips` package is not installed. Install the extra: "
            "pip install '.[perceptual]'. 'dists' needs no extra and can be used "
            "on its own in the meantime."
        )
    from torchmetrics.functional.image import learned_perceptual_image_patch_similarity

    return learned_perceptual_image_patch_similarity


def _dists_fn() -> Callable:
    """Import DISTS on demand. No availability gate — torchvision is a core dep."""
    from torchmetrics.functional.image import deep_image_structure_and_texture_similarity

    return deep_image_structure_and_texture_similarity


def perceptual_score(
    name: str, sr: torch.Tensor, hr: torch.Tensor, lpips_net: str = "alex"
) -> torch.Tensor:
    """Score one perceptual metric over a batch, reduced to a 0-dim tensor.

    Args:
        name: ``'lpips'`` or ``'dists'``.
        sr: Reconstruction, ``(B, 3, H, W)`` RGB float in ``[0, 1]``.
        hr: Reference, same shape and range.
        lpips_net: LPIPS backbone — ``'alex'``, ``'vgg'`` or ``'squeeze'``.
            Ignored by DISTS. A LPIPS figure is comparable only to one computed
            under the same backbone, the same way an SSIM figure is comparable
            only within one ``ssim_impl``.

    Returns:
        0-dim tensor, batch-meaned.

    Raises:
        ValueError: If ``name`` is not a supported metric.
        ImportError: If ``name`` is ``'lpips'`` and the extra is not installed.
    """
    if name not in PERCEPTUAL_METRICS:
        raise ValueError(
            f"perceptual metric must be one of {sorted(PERCEPTUAL_METRICS)}; got {name!r}"
        )
    if name == "lpips":
        # normalize=True declares "inputs are [0,1]". The default False means
        # "already [-1,1]" and would score a squashed range without erroring.
        # reduction is passed explicitly because DISTS below defaults to None
        # while this defaults to 'mean' — relying on either gives two shapes.
        return _lpips_fn()(sr, hr, net_type=lpips_net, reduction="mean", normalize=True)
    return _dists_fn()(sr, hr, reduction="mean")
