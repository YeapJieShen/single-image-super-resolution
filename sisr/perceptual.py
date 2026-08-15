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

**Both backbones are memoised here, module-level, keyed by
``(metric, lpips_net, device, dtype)``.** torchmetrics' *functional* API builds
a fresh network per call, which dominated the metric entirely: measured on CPU
at 1x3x256x256, 605.0 ms/call of which 27.3 ms was LPIPS itself, and
1716.1 ms/call of which 386.8 ms was DISTS itself. One validation cycle of the
shipped adversarial template scores ~219 images per metric, so that was ~7
minutes of pure construction per cycle, paid again at step 0 through
``num_sanity_val_steps``. The cache is a module global rather than an attribute
on the Lightning module for the same reason the VGG19 criterion is held outside
the module tree: an ``nn.Module`` attribute would put ~230 MB of AlexNet (or
~528 MB of VGG16) into every ``state_dict`` this project writes.
"""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class _ResettableMetric(Protocol):
    """Structural type for the "lpips" branch's net.

    Whatever ``_lpips_metric_cls`` returns (the real torchmetrics class, or a
    test double standing in for it) is matched by shape rather than base
    class, so the seam stays swappable.
    """

    def reset(self) -> None: ...


#: Metric name -> is-lower-better. Consumed by the checkpoint-monitor direction
#: check: PSNR/SSIM are higher-better and SRCheckpoint defaults to mode='max',
#: so monitoring one of these at the default would keep the *worst* model.
PERCEPTUAL_METRICS: dict[str, bool] = {"lpips": True, "dists": True}

#: Memoised backbones, keyed by ``(name, lpips_net, device, dtype)``. Populated by
#: :func:`_cached_net`; never read directly.
_NET_CACHE: dict[tuple[str, str, torch.device, torch.dtype], torch.nn.Module] = {}


def _lpips_metric_cls() -> type:
    """Import LPIPS on demand, turning a missing extra into an actionable error.

    Indirected through a function (rather than a module-level import) so the
    import cost is only paid by runs that ask for it — this module is reachable
    from spawned DataLoader workers, which pay every import twice on Windows.

    Returns the *modular* metric class, not the functional entry point: it holds
    its ``_NoTrainLpips`` backbone as ``self.net``, built once at construction,
    which is what makes it cacheable.
    """
    from torchmetrics.utilities.imports import _LPIPS_AVAILABLE

    if not _LPIPS_AVAILABLE:
        raise ImportError(
            "perceptual_metrics includes 'lpips', but torchmetrics reports the "
            "`lpips` package is not installed. Install the extra: "
            "pip install '.[perceptual]'. 'dists' needs no extra and can be used "
            "on its own in the meantime."
        )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    return LearnedPerceptualImagePatchSimilarity


def _dists_net_cls() -> type:
    """Import DISTS's backbone on demand. No availability gate — torchvision is a core dep.

    The bare network, not torchmetrics' ``DeepImageStructureAndTextureSimilarity``:
    that metric's ``update`` calls ``_dists_update``, which constructs a whole
    ``DISTSNetwork`` per call, so the modular class is exactly as expensive as
    the functional one and cannot be cached (verified on torchmetrics 1.9.0).
    LPIPS's modular class does hold its backbone, hence the asymmetry.
    """
    from torchmetrics.functional.image.dists import DISTSNetwork

    return DISTSNetwork


def _cached_net(
    name: str, lpips_net: str, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
    """Return the memoised backbone for one metric on one device/dtype.

    Built in ``eval()`` with every parameter detached from autograd, so a cached
    backbone can neither train nor leak gradients into the model being scored.
    Construction is forced out of ``inference_mode`` — Lightning's validation
    loop runs inside it, and weights allocated there would be inference tensors
    baked into a cache that outlives the block.

    Args:
        name: ``'lpips'`` or ``'dists'``.
        lpips_net: LPIPS backbone name; ignored (and normalised out of the cache
            key) for DISTS.
        device: Device the scored tensors live on.
        dtype: Dtype the scored tensors carry.

    Returns:
        The cached ``nn.Module``, already on ``device``/``dtype``.
    """
    key = (name, lpips_net if name == "lpips" else "", device, dtype)
    net = _NET_CACHE.get(key)
    if net is None:
        with torch.inference_mode(False):
            if name == "lpips":
                # normalize=True declares "inputs are [0,1]". The default False means
                # "already [-1,1]" and would score a squashed range without erroring.
                # reduction is passed explicitly because DISTS defaults to None while
                # this defaults to 'mean' — relying on either gives two shapes.
                net = _lpips_metric_cls()(net_type=lpips_net, reduction="mean", normalize=True)
            else:
                net = _dists_net_cls()()
            net = net.to(device=device, dtype=dtype).eval().requires_grad_(False)
        _NET_CACHE[key] = net
    return net


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
    net = _cached_net(name, lpips_net, sr.device, sr.dtype)
    with torch.no_grad():
        if name == "lpips":
            score = net(sr, hr)
            # The metric object is reused, so its accumulators must not be: LPIPS
            # keeps a list state and would retain one tensor per call for the whole
            # run. Only the per-call value returned above is ever read.
            assert isinstance(net, _ResettableMetric)  # the "lpips" branch of _cached_net
            net.reset()
            return score
        # require_grad=False matches what the functional path passes for the
        # no-grad tensors a val loop produces, and is unconditional here because
        # nothing may build a graph through a scoring metric.
        return net(sr, hr, require_grad=False).mean()
