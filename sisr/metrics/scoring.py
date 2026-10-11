"""One owner for "score an aligned ``(sr_rgb, hr_rgb)`` pair under an ``SREvalConfig``".

Before this module the answer to "how does this project score a reconstruction?"
was spread across two files and reachable only through private methods.
``SRLightning`` exposed ``_build_metric_tensors`` / ``_mean_psnr`` /
``_mean_ssim`` / ``_mean_perceptual`` as a de-facto interface and
``BenchmarkImageLogger`` reached through all four — its own comment conceded
the reach, saying "the colorspace split has no public seam". Four separate
decisions had no single home:

* the **crop-border slice**, written once in ``validation_step`` and again in
  ``BenchmarkImageLogger._collect_batch``;
* the **colorspace split** into scored key tensors;
* the **PSNR reduction** — per-image then batch mean, never batch-pooled;
* which SSIM ``ssim_impl`` names.

And the **tag grammar** ``{family}/{scope}/{key}`` was rebuilt from f-strings in
four places, so nothing stopped ``psnr/val/Y`` and ``psnr/Set5/Y`` drifting apart.

Everything above now lives here, and the grammar has exactly one implementation
in :func:`metric_tag`. Callers supply the *scope* — ``"val"`` for the validation
loop, a dataset name for the benchmark loop — and nothing else.

This module deliberately depends on ``SREvalConfig`` and tensors only: no
``SRLightning``, no ``Trainer``, no datamodule. Scoring is testable on two
tensors and a config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torchmetrics.functional.image

from ..colorspace import rgb_to_ycbcr, rgb_to_ycbcr_studio
from .perceptual import perceptual_score
from .ssim import daala_ssim, quantize_u8

if TYPE_CHECKING:  # `SREvalConfig` is referenced in annotations only, and a
    # runtime import would make this package depend on `sisr.training` — which
    # imports back into `sisr.metrics`, closing a cycle.
    from ..training.config import SREvalConfig

__all__ = ["SRScorer", "Scores", "metric_tag"]


def _scale_suffix(scale: int | None) -> str:
    return "" if scale is None else f"/x{scale}"


def metric_tag(family: str, scope: str, key: str, scale: int | None = None) -> str:
    """Build a metric tag: ``{family}/{scope}/{key}``, plus ``/x{scale}`` when a scale is given.

    The single implementation of the tag grammar. ``psnr/val/Y`` and
    ``psnr/Set5/Y`` differ only in ``scope``, so routing both through here is
    what stops the validation and benchmark hierarchies drifting apart.

    Args:
        family: Metric family — ``'psnr'``, ``'ssim'``, or a perceptual name.
        scope: ``'val'`` for the validation loop, else a benchmark dataset name.
        key: Colorspace or channel key, e.g. ``'RGB'`` or ``'Y'``.
        scale: The scale this value was measured at, for a multi-scale run's
            per-scale validation tags (``psnr/val/Y/x2``). ``None`` is the plain tag --
            for a multi-scale run, the mean over its scales. A benchmark set instead
            carries its scale in its *name* (``Set5_x2``), so it never needs this.

    Returns:
        The TensorBoard tag.
    """
    return f"{family}/{scope}/{key}{_scale_suffix(scale)}"


#: A metric value as it travels to ``pl_module.log``. The validation path
#: produces 0-dim tensors straight from the metric functions; the benchmark
#: path means its per-image buffer first and so produces plain floats. Both are
#: legal ``log`` arguments, and ``Scores`` is only a carrier -- it never does
#: arithmetic on them -- so it accepts either rather than forcing one side to
#: convert for the type checker's benefit.
ScoreValue = torch.Tensor | float


def perceptual_tag(name: str, scope: str, scale: int | None = None) -> str:
    """Build a perceptual metric's tag: ``{name}/{scope}``, plus ``/x{scale}`` when given.

    The second rule of the grammar, and deliberately a different shape:
    LPIPS/DISTS carry no colorspace key, so they get their own family root
    rather than a key slot under the PSNR/SSIM scheme. That shape is already
    published, so it is preserved here rather than normalised -- which is
    exactly why it needs an owner too. A caller re-deriving "the tag grammar"
    from :func:`metric_tag` alone silently gets this family wrong.

    Args:
        name: Perceptual metric name, e.g. ``'lpips'``.
        scope: ``'val'`` for the validation loop, else a benchmark dataset name.
        scale: See :func:`metric_tag`.

    Returns:
        The TensorBoard tag.
    """
    return f"{name}/{scope}{_scale_suffix(scale)}"


def expected_tags(eval_config: SREvalConfig, scope: str) -> list[str]:
    """Every tag a run under *eval_config* will log at *scope*.

    The values-free half of the grammar: :meth:`Scores.tagged` answers "which
    tags exist" **with** numbers, this answers it without any. Both derive from
    :func:`metric_tag` and :func:`perceptual_tag`, so a caller that needs to
    know the tag namespace before a single batch has been scored -- seeding
    TensorBoard's HParams columns, or validating a checkpoint monitor -- asks
    the owner instead of re-deriving the grammar from f-strings.

    That re-derivation is not hypothetical: the checkpoint monitor validator,
    whose whole job is knowing which tags will exist, had its own copy. When
    the two disagree the failure is silent -- a correct monitor is rejected, or
    an incorrect one accepted and the run keeps its worst checkpoint.

    Args:
        eval_config: The evaluation settings the run will score under.
        scope: ``'val'`` for the validation loop, else a benchmark dataset name.

    Returns:
        Every tag, PSNR then SSIM then perceptual, in the order
        :meth:`Scores.tagged` produces them.
    """
    tags = [metric_tag("psnr", scope, k) for k in eval_config.psnr_keys]
    tags += [metric_tag("ssim", scope, k) for k in eval_config.ssim_keys]
    tags += [perceptual_tag(name, scope) for name in eval_config.perceptual_keys]
    return tags


class Scores:
    """Metric values for one scored pair, before any tag prefixing.

    Attributes:
        psnr: ``key -> value``, over ``eval_config.psnr_keys``.
        ssim: ``key -> value``, over ``eval_config.ssim_keys``.
        perceptual: ``name -> value``, over ``eval_config.perceptual_keys``.
    """

    __slots__ = ("psnr", "ssim", "perceptual")

    def __init__(
        self,
        psnr: dict[str, ScoreValue],
        ssim: dict[str, ScoreValue],
        perceptual: dict[str, ScoreValue],
    ) -> None:
        self.psnr = psnr
        self.ssim = ssim
        self.perceptual = perceptual

    def tagged(self, scope: str, scale: int | None = None) -> dict[str, ScoreValue]:
        """Flatten to ``tag -> value`` under ``scope``.

        Perceptual metrics carry no colorspace key, so they tag as
        ``{name}/{scope}`` rather than ``{family}/{scope}/{key}`` — the shape
        already published and therefore preserved here.

        Args:
            scope: ``'val'`` or a benchmark dataset name.
            scale: Suffix every tag with the scale it was measured at (see :func:`metric_tag`).

        Returns:
            Every metric under its full tag.
        """
        tags = {metric_tag("psnr", scope, k, scale): v for k, v in self.psnr.items()}
        tags |= {metric_tag("ssim", scope, k, scale): v for k, v in self.ssim.items()}
        tags |= {perceptual_tag(name, scope, scale): v for name, v in self.perceptual.items()}
        return tags


class SRScorer:
    """Scores aligned ``(sr_rgb, hr_rgb)`` pairs under one ``SREvalConfig``.

    Built from the config alone, so it can be constructed and exercised without
    a Lightning module, a trainer or a datamodule.

    Args:
        eval_config: The evaluation settings to score under.
    """

    def __init__(self, eval_config: SREvalConfig) -> None:
        self.eval_config = eval_config

    def crop(self, *tensors: torch.Tensor, scale: int | None = None) -> tuple[torch.Tensor, ...]:
        """Apply ``crop_border`` to each tensor, or pass them through unchanged.

        The single implementation of the border slice. Both scoring paths took
        their own copy of this before; a divergence between them would have
        silently changed what every published figure was computed on.

        Args:
            *tensors: ``(B, C, H, W)`` tensors to crop identically.
            scale: The scale these tensors were produced at. Used **only** when
                ``crop_border`` is ``None``, which is what a multi-scale run keeps (there
                is no one border to resolve): the border is then this scale, the field
                convention, so each scale is scored on its own border.

        Returns:
            The cropped tensors, in the order given.

        Raises:
            ValueError: If ``crop_border`` is ``None`` and no ``scale`` was given. A
                single-scale run never reaches this: ``SRLightning`` resolves the
                derive-from-scale sentinel to an int at construction, so ``None`` here
                means a config that never passed through it; scoring against an
                unresolved border would silently pick a different region than the run
                intends.
                Also if it is negative -- reachable only past
                ``SREvalConfig.__post_init__``, via the sentinel resolution's
                direct assignment -- or if twice the border would consume an
                image's shorter axis. The oversized case cannot be caught at
                config time because no image is in scope there, so it is caught
                here, where the tensor is, and names both the border and the
                size rather than surfacing as a torch shape error.
        """
        n = self.eval_config.crop_border
        if n is None:
            n = scale
        if n is None:
            raise ValueError(
                "eval_config.crop_border is still None at scoring time. It is a sentinel "
                "meaning 'crop the model's scale', resolved by SRLightning at construction "
                "-- build the scorer from a module's eval_config, or set an explicit int."
            )
        if n < 0:
            raise ValueError(
                f"eval_config.crop_border is {n} at scoring time. SREvalConfig rejects a "
                "negative border at construction, but SRLightning ASSIGNS this field when it "
                "resolves the derive-from-scale sentinel, which bypasses that check -- so a "
                "non-positive `scale` lands here. Scoring would silently return the FULL "
                "image, because 'no crop' and 'a negative crop' are the same branch."
            )
        if n == 0:
            return tensors
        for t in tensors:
            h, w = t.shape[-2:]
            if 2 * n >= min(h, w):
                raise ValueError(
                    f"eval_config.crop_border={n} removes {2 * n} px from each axis of a "
                    f"{h}x{w} image, leaving nothing to score. Lower crop_border, or drop "
                    "the images smaller than it from the evaluation set."
                )
        return tuple(t[..., n:-n, n:-n] for t in tensors)

    def metric_tensors(
        self, sr: torch.Tensor, hr: torch.Tensor, is_grey: torch.Tensor | None = None
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Pre-compute ``(sr, hr)`` pairs for every tracked PSNR/SSIM key.

        Computed over the union of ``psnr_keys`` and ``ssim_keys`` so a
        requested colorspace conversion happens at most once per call regardless
        of how many metrics consume it. ``Y`` / ``Cb`` / ``Cr`` / ``YCbCr`` go
        through ``rgb_to_ycbcr_studio`` — BT.601 **studio** range, the
        literature's PSNR/SSIM convention — not the full-range ``rgb_to_ycbcr``
        the processors train in (see :mod:`sisr.colorspace`). ``RGB``/``R``/
        ``G``/``B`` are untouched either way.

        Args:
            sr: Reconstruction, ``(B, 3, H, W)`` RGB float in ``[0, 1]``.
            hr: Reference, same shape.
            is_grey: ``(B,)`` bool, per image: the reference file is single-channel. Required
                when ``grey_plane_scoring`` is on, ignored otherwise.

        Returns:
            ``key -> (sr_key, hr_key)``.

        Raises:
            ValueError: If ``grey_plane_scoring`` is on and ``is_grey`` is ``None``.
        """
        keys = set(self.eval_config.psnr_keys) | set(self.eval_config.ssim_keys)
        tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        if keys & {"RGB", "R", "G", "B"}:
            tensors["RGB"] = (sr, hr)
            tensors["R"] = (sr[:, 0:1], hr[:, 0:1])
            tensors["G"] = (sr[:, 1:2], hr[:, 1:2])
            tensors["B"] = (sr[:, 2:3], hr[:, 2:3])

        if keys & {"YCbCr", "Y", "Cb", "Cr"}:
            sr_ycc = rgb_to_ycbcr_studio(sr)
            hr_ycc = rgb_to_ycbcr_studio(hr)
            if self.eval_config.grey_plane_scoring:
                if is_grey is None:
                    raise ValueError(
                        "eval_config.grey_plane_scoring is on but no is_grey flag reached "
                        "scoring: set yield_is_grey: true on every eval dataset."
                    )
                # A grey file's raw plane is its full-range Y (the weights sum to 1).
                grey = is_grey.to(hr.device).view(-1, 1, 1, 1)
                sr_ycc = torch.cat(
                    [torch.where(grey, rgb_to_ycbcr(sr)[:, :1], sr_ycc[:, :1]), sr_ycc[:, 1:]], 1
                )
                hr_ycc = torch.cat(
                    [torch.where(grey, rgb_to_ycbcr(hr)[:, :1], hr_ycc[:, :1]), hr_ycc[:, 1:]], 1
                )
            tensors["YCbCr"] = (sr_ycc, hr_ycc)
            tensors["Y"] = (sr_ycc[:, 0:1], hr_ycc[:, 0:1])
            tensors["Cb"] = (sr_ycc[:, 1:2], hr_ycc[:, 1:2])
            tensors["Cr"] = (sr_ycc[:, 2:3], hr_ycc[:, 2:3])

        return tensors

    def _rounded(
        self, pair: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The pair on the 8-bit grid when ``round_to_uint8`` is set, else as given (PSNR only)."""
        if not self.eval_config.round_to_uint8:
            return pair
        return tuple(quantize_u8(t) / 255.0 for t in pair)  # type: ignore[return-value]

    @staticmethod
    def psnr(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Mean of the per-image PSNRs across the batch (SR-standard reduction).

        Scores each image independently (``dim=(1, 2, 3)``) before the batch
        mean, so the result is invariant to the val ``batch_size``. This differs
        from pooling the whole batch into one PSNR — the deviation a stateful
        ``PeakSignalNoiseRatio`` with default ``dim`` would introduce. The two
        agree only when every image in the batch is equally wrong, which is why
        a batch-of-one caller cannot detect the difference.

        Args:
            sr: Reconstruction, ``(B, C, H, W)``.
            hr: Reference, same shape.

        Returns:
            0-dim tensor.
        """
        return torchmetrics.functional.image.peak_signal_noise_ratio(
            sr, hr, data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean"
        )

    def ssim(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Mean SSIM under the configured implementation.

        The single decision point for what ``ssim_impl`` names. Both
        implementations reduce per-image then mean over the batch.

        Args:
            sr: Reconstruction, ``(B, C, H, W)`` float in ``[0, 1]``.
            hr: Reference, same shape.

        Returns:
            0-dim tensor.
        """
        if self.eval_config.ssim_impl == "daala":
            return daala_ssim(sr, hr)
        # torchmetrics widens the return to a tuple only when return_full_image or
        # return_contrast_sensitivity is set; both default False, so this call site
        # always yields the scalar.
        return cast(
            torch.Tensor,
            torchmetrics.functional.image.structural_similarity_index_measure(
                sr, hr, data_range=1.0
            ),
        )

    def perceptual(self, name: str, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Mean perceptual score under the configured backbone.

        Args:
            name: Any key of :data:`~sisr.metrics.perceptual.PERCEPTUAL_METRICS`.
            sr: Reconstruction, ``(B, 3, H, W)`` RGB float in ``[0, 1]``.
            hr: Reference, same shape. Ignored by the no-reference metrics, which
                score the same cropped ``sr`` the paired metrics see.

        Returns:
            0-dim tensor.
        """
        return perceptual_score(name, sr, hr, lpips_net=self.eval_config.lpips_net)

    def score(
        self,
        sr_rgb: torch.Tensor,
        hr_rgb: torch.Tensor,
        *,
        crop: bool = True,
        scale: int | None = None,
        is_grey: torch.Tensor | None = None,
    ) -> Scores:
        """Score one aligned pair across every metric the config requests.

        Args:
            sr_rgb: Reconstruction, ``(B, 3, H, W)`` RGB float in ``[0, 1]``.
            hr_rgb: Reference, same shape.
            crop: Apply ``crop_border`` first. Pass ``False`` only when the
                caller has already cropped — scoring an uncropped pair silently
                changes every number.
            scale: The pair's scale, for :meth:`crop` when the border is derived from it.
            is_grey: Per-image single-channel-file flags, see :meth:`metric_tensors`.

        Returns:
            The metric values, untagged.
        """
        if crop:
            sr_rgb, hr_rgb = self.crop(sr_rgb, hr_rgb, scale=scale)

        tensors = self.metric_tensors(sr_rgb, hr_rgb, is_grey)
        return Scores(
            psnr={k: self.psnr(*self._rounded(tensors[k])) for k in self.eval_config.psnr_keys},
            ssim={k: self.ssim(*tensors[k]) for k in self.eval_config.ssim_keys},
            perceptual={
                name: self.perceptual(name, sr_rgb, hr_rgb)
                for name in self.eval_config.perceptual_keys
            },
        )
