"""Per-architecture training and evaluation configuration dataclasses.

Split by lifecycle: ``SRTrainingConfig`` controls ``cli fit`` (per-Conv2d LRs,
paper-init knobs, the TensorBoard graph's example input); ``SREvalConfig``
controls validation/test scoring only (border exclusion, metric colorspaces).

Per-architecture defaults live in subclasses next to the model code (e.g.
``sisr.models.srcnn.SRCNNTrainingConfig``); YAML picks one via ``class_path``
on ``model.training_config`` / ``model.eval_config``.

The model's training colorspace is not a field here — it is the choice of
processor (see ``sisr.processors``).

``validate_against`` / ``psnr_keys`` / ``ssim_keys`` are the config half of the
correlated-field validation seam: ``num_channels`` (model) and ``class_path``
(processor) live in sibling objects no single ``__post_init__`` can see across,
so the cross-object check runs once both exist, orchestrated by
``SRLightning.__init__``.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch

from ..metrics.perceptual import PERCEPTUAL_METRICS
from ..models.base import SRModel
from ..processors.base import SRProcessor

# Maps a colorspace to its sub-channel names, expanded only when separate_psnr=True;
# single-channel entries map to () so a bare channel name is itself a legal
# psnr_channels/ssim_channels entry. Doubles as the allowlist for both metric
# families, so PSNR and SSIM cannot diverge on which names are legal.
_CHANNEL_SUBNAMES: dict[str, tuple[str, ...]] = {
    "RGB": ("R", "G", "B"),
    "YCbCr": ("Y", "Cb", "Cr"),
    "R": (),
    "G": (),
    "B": (),
    "Y": (),
    "Cb": (),
    "Cr": (),
}


@dataclass
class SRTrainingConfig:
    """How to train the SR model — affects optimizer setup and weight init.

    Args:
        layer_lrs: Absolute per-``Conv2d`` LRs, one entry per ``Conv2d`` in
            module-traversal order. ``None`` (default) uses the optimizer's
            base ``lr`` uniformly. Only valid where every trainable parameter
            lives in a ``Conv2d`` — no BatchNorm / PReLU;
            ``SRLightning.configure_optimizers`` raises ``ValueError``
            otherwise. An entry is either a **float**, applying to that
            layer's weight *and* bias together, or a
            ``[weight_lr, bias_lr]`` **pair** splitting them — which is what
            a Caffe prototxt's two per-layer ``param`` blocks express, and
            the only way to state a bias schedule that differs from its
            layer's weight schedule. LRs are absolute either way, so there
            is no base rate to be relative to.

        example_input_shape: One sample's shape *excluding* batch (e.g.
            ``(1, 33, 33)``). Sets ``example_input_array`` so TensorBoard can
            capture the graph, and gives :meth:`validate_against` a probe.

        init_strategy: ``'paper'`` runs ``SRModel.reset_parameters`` from
            ``SRLightning``'s constructor; ``'default'`` uses PyTorch's.
            Subclasses pin a paper-faithful default.

        init_mean: Gaussian mean for ``init_strategy='paper'``. **Not
            paper-derived** — this base has no single paper; architectures with
            one override it.

        init_std: Gaussian std for ``init_strategy='paper'``. Not paper-derived
            either — see ``init_mean``.

        scale: The model's upscaling factor, for provenance metadata and for
            cross-checking the model's own ``scale`` hparam. **Required for any
            run that writes an artifact** on an architecture carrying no
            ``scale`` hparam: :func:`~sisr.training.metadata.build_metadata`
            refuses rather than record a null, because a file handed to a
            stranger must state the factor — for a pre-upsampled architecture
            it is what they resize the input by. Deliberately not inferred from
            the datamodule: that attribute is outside the ``SRDataset``
            contract (``PredictDataset`` has none), whereas per-paper knobs
            already live here.

        compile_backend: A ``torch._dynamo`` backend to compile the
            training-mode forward with; ``None`` (default) trains eager. **Only
            ``training_step`` is compiled** — validation/test/predict stay eager,
            which the widely varying benchmark image sizes require. An
            unrecognized name raises at ``SRLightning`` construction; a
            recognized name with a missing toolchain (e.g. ``'inductor'``
            without Triton) would fail on first call, so
            ``SRLightning.on_fit_start`` forces it early with a warm-up forward
            rather than crash mid-run. Measured: ``'inductor'`` is the only
            backend that pays, by a mid-teens percent; ``'cudagraphs'`` is
            within noise (it captures the forward only) and ``'aot_eager'`` is
            slower; on SRCNN eager wins outright. **No backend is bit-identical
            to eager on SRResNet**, so a compiled run does not reproduce an
            eager one. Staying at ``None`` is the standing recommendation.

        scalar_logging: Which per-step scalars a training step records.
            ``self.log`` costs real host time on **every** step, and
            ``log_every_n_steps`` gates the *flush*, not the accumulate, so the
            existing cadence knob does not touch it. Those scalars are the run's
            evidence, so this is opt-in and never silently applied.

            - ``'all'`` (default) -- unchanged behaviour, every scalar below.
            - ``'essential'`` -- keeps only the objective each optimiser
              actually steps on: ``loss/train``, plus ``loss/train/d`` under
              manual optimization. **Stops recording:** ``loss/train/content``
              and ``loss/train/adv`` on the adversarial path, and every
              per-term tag a composite criterion exposes through
              ``last_terms`` (``loss/train/<term>``). Validation scalars are
              untouched -- they are per-cycle, not per-step.

            **What this is measured to be worth, so the trade is visible
            here rather than inferred.** On an RTX 5060 Laptop: one ``self.log``
            call costs **1.132 ms on a 6.167 ms SRCNN step (18.4%)**, and is
            **below the noise floor of a 36 ms SRResNet step**. But that 1.132 ms
            is ``loss/train`` itself — the objective, which ``'essential'``
            keeps — so on both *shipped* configs this knob recovers nothing
            measurable. It pays only for a criterion with many terms, where the
            decomposition is several extra calls per step. Treat it as a lever
            for that case, not as a general speedup.

            **Two runs that logged different scalar sets are not
            interchangeable**, which is why this lives on the training config
            and so reaches ``hparams`` and the checkpoint, rather than on a
            callback.
        compile_mode: ``torch.compile``'s ``mode``, applied alongside
            ``compile_backend``. ``None`` (default) leaves inductor on its own
            default. **Requires ``compile_backend='inductor'``** — ``mode`` is
            an inductor setting, and torch forwards it to any other backend as
            a keyword its compiler does not accept, failing on the first
            compiled call; this refuses at construction instead. An
            unrecognized mode raises there too, from torch. **Which mode wins
            does not hold across precisions**: ``'max-autotune'`` beat
            ``'reduce-overhead'`` under bf16 and lost under fp32 — measure per
            configuration, never assume.
    """

    layer_lrs: list[float | list[float]] | None = None
    example_input_shape: tuple[int, ...] | None = None
    init_strategy: Literal["default", "paper"] = "default"
    init_mean: float = 0.0
    init_std: float = 0.01
    scale: int | None = None
    compile_backend: str | None = None
    compile_mode: str | None = None
    scalar_logging: Literal["all", "essential"] = "all"

    def __post_init__(self) -> None:
        """Reject a compile mode that nothing will apply, and values that mean nothing.

        Every check here names the YAML key to fix, matching
        :class:`SREvalConfig`'s style. They exist because each of these fields
        reached something that accepted it silently: a negative ``scale``
        resolves ``eval_config.crop_border`` to a negative border and scores
        uncropped, a negative ``layer_lrs`` entry trains that layer in the
        wrong direction, and ``init_strategy`` dispatches on ``== "paper"``, so
        any other spelling means "default" with nothing saying so.

        ``init_mean`` is deliberately unconstrained -- a negative mean is a
        legitimate gaussian.

        Raises:
            ValueError: If ``compile_mode`` is set without
                ``compile_backend='inductor'``; if ``scale`` is set and is not
                a positive int; if any ``layer_lrs`` entry (or either half of
                a ``[weight_lr, bias_lr]`` pair) is not positive; if
                ``init_std`` is not positive; or if ``init_strategy`` is
                neither ``'default'`` nor ``'paper'``.
        """
        if self.scale is not None and (
            not isinstance(self.scale, int) or isinstance(self.scale, bool) or self.scale < 1
        ):
            raise ValueError(
                f"training_config.scale must be a positive int; got {self.scale!r} "
                f"({type(self.scale).__name__}). "
                "It is not only provenance: for an architecture that declares no `scale` "
                "hparam of its own (SRCNN, deliberately -- it is resolution-preserving) "
                "nothing else checks it, and eval_config.crop_border derives from it, so a "
                "non-positive scale becomes a non-positive border and the metric is computed "
                "on the uncropped image. Fix model.training_config.init_args.scale in your YAML."
            )

        if self.layer_lrs is not None:
            for i, lr in enumerate(self.layer_lrs):
                pair = [lr] if isinstance(lr, int | float) else list(lr)
                if any(v <= 0 for v in pair):
                    raise ValueError(
                        f"training_config.layer_lrs[{i}]={lr!r} must be positive "
                        "(every entry, and both halves of a [weight_lr, bias_lr] pair). "
                        "These are absolute learning rates handed straight to the optimizer: "
                        "a negative one ascends the loss for that layer and a zero freezes it "
                        "silently. Only the list's LENGTH is checked when the optimizer is "
                        "built. Fix model.training_config.init_args.layer_lrs in your YAML."
                    )

        if self.init_std <= 0:
            raise ValueError(
                f"training_config.init_std must be positive; got {self.init_std}. "
                "It reaches torch.nn.init.normal_ as the gaussian's standard deviation. "
                "(init_mean is unconstrained -- a negative mean is legitimate.) "
                "Fix model.training_config.init_args.init_std in your YAML."
            )

        if self.init_strategy not in ("default", "paper"):
            raise ValueError(
                f"training_config.init_strategy must be 'default' or 'paper'; got "
                f"{self.init_strategy!r}. The dispatch tests `== 'paper'`, so any other "
                "spelling -- 'Paper' included -- silently selects PyTorch's default init "
                "while the config reads as though the paper's runs. This is a "
                "paper-reproduction framework, so that difference is the whole point of the "
                "field. The Literal annotation is enforced by jsonargparse at the CLI and by "
                "nothing in Python, so a directly-constructed config had no guard at all. "
                "Fix model.training_config.init_args.init_strategy in your YAML."
            )

        if self.scalar_logging not in ("all", "essential"):
            raise ValueError(
                f"training_config.scalar_logging must be 'all' or 'essential'; got "
                f"{self.scalar_logging!r}. Fix "
                f"model.training_config.init_args.scalar_logging in your YAML."
            )
        if self.compile_mode is not None and self.compile_backend != "inductor":
            raise ValueError(
                f"compile_mode={self.compile_mode!r} needs compile_backend='inductor'; got "
                f"compile_backend={self.compile_backend!r}. `mode` is an inductor setting: "
                "with no backend nothing is compiled and the mode is dead config, and with "
                "another backend torch passes `mode` to a compiler function that does not "
                "take it, which fails on the first compiled call rather than at startup."
            )

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Validate this config against the model/processor it will pair with.

        Universal checks: ``self.scale`` must agree with the model's ``'scale'``
        hparam when both exist (either absent skips, rather than guessing); and
        when ``example_input_shape`` is set its channel dimension must equal
        ``processor.model_channels`` and a ``no_grad`` forward of the real
        model must succeed. The probe exercises the actual ``nn.Module``, so it
        cannot go stale as the architecture evolves.

        Subclasses add architecture-specific correlation checks so a mismatched
        pairing raises an actionable message *before* it would surface as a raw
        shape mismatch from this probe; they call ``super()`` to keep these.

        Args:
            model: The constructed :class:`~sisr.models.base.SRModel`.
            processor: The :class:`~sisr.processors.base.SRProcessor` paired
                with ``model``.

        Raises:
            ValueError: If ``self.scale`` disagrees with the model's ``scale``
                hparam, or ``example_input_shape``'s channel dimension does not
                match ``processor.model_channels``.
        """
        if self.scale is not None and "scale" in model.hparams:
            model_scale = model.hparams["scale"]
            if self.scale != model_scale:
                raise ValueError(
                    f"training_config.scale={self.scale} does not match "
                    f"model.hparams['scale']={model_scale}. Fix training_config.scale "
                    f"or the model's scale hyperparameter — they must agree."
                )

        if self.example_input_shape is None:
            return
        channels = self.example_input_shape[0]
        if channels != processor.model_channels:
            raise ValueError(
                f"training_config.example_input_shape has {channels} channel(s) "
                f"(position 0), but {type(processor).__name__}.model_channels="
                f"{processor.model_channels}. Fix example_input_shape[0] or pick "
                f"a processor whose model_channels matches the model's actual "
                f"input/output channel count."
            )
        dummy = torch.zeros(1, *self.example_input_shape)
        with torch.no_grad():
            model(dummy)


@dataclass
class SREvalConfig:
    """How to compute validation/test metrics — affects scoring only, not training.

    Args:
        crop_border: Border pixels excluded on each edge before PSNR / SSIM.
            **``None`` means the SR field's convention: the model's own
            ``scale``**, resolved once by ``SRLightning`` at construction — the
            SRCNN authors' released demo code shaves exactly ``scale`` per side.
            An explicit int is an intent and always wins, ``0`` included. The
            base default stays ``0``; the per-architecture subclasses use
            ``None``, so their border can no longer be right at one scale and
            silently wrong at another.

        psnr_channels: Colorspaces or bare single channels PSNR is reported
            for.  Supported values are ``'RGB'``, ``'YCbCr'``, and any
            individual channel name (``'R'``, ``'G'``, ``'B'``, ``'Y'``,
            ``'Cb'``, ``'Cr'``).  Multiple entries are allowed (e.g.
            ``['RGB', 'Y']`` produces ``psnr/val/RGB`` and ``psnr/val/Y`` —
            the paper-comparable Y-only metric, without YCbCr's
            three-channel aggregate diluting it with smoother chroma planes).
            ``Y``/``Cb``/``Cr``/``YCbCr`` are scored in BT.601 studio range
            (the literature's convention); ``RGB``/``R``/``G``/``B`` are
            unaffected.

        separate_psnr: When ``True``, also reports PSNR for each individual
            channel within each requested colorspace (e.g. ``'RGB'`` adds
            ``psnr/val/R`` / ``psnr/val/G`` / ``psnr/val/B`` alongside
            the aggregate ``psnr/val/RGB``).

        ssim_channels: Colorspaces or bare single channels SSIM is reported
            for. Same allowlist and studio-range treatment as
            ``psnr_channels``; no ``separate_ssim`` counterpart to
            ``separate_psnr`` exists because per-channel R/G/B or Cb/Cr SSIM
            has no established convention to reproduce. Defaults to
            ``['RGB', 'Y']`` (not just ``['RGB']`` like ``psnr_channels``) so
            the paper-comparable Y-SSIM ships out of the box — unlike PSNR,
            RGB-SSIM cannot be corrected to Y-SSIM after the fact by a
            constant offset, so there is no reason to gate it behind an
            architecture-specific subclass.

        ssim_impl: Which SSIM to compute. ``'wang'`` (default) is the
            field-standard fixed 11x11 gaussian, sigma 1.5 — what
            ``torchmetrics``, MATLAB and BasicSR compute, and therefore what
            most SR papers report. ``'daala'`` is the daala package's
            resolution-adaptive variant (sigma scales with image height), the
            convention Ledig et al. actually used; see :mod:`sisr.metrics.ssim`.
            Switching is **in place** — the ``ssim/...`` metric names do not
            change, so a figure is comparable only to one computed under the
            same setting. The value is recorded in ``hparams`` and in every
            artifact's ``sisr_meta`` so any checkpoint can be traced back.

        perceptual_metrics: Perceptual metrics to report, from ``'lpips'`` and
            ``'dists'``. Empty by default, so an architecture that never asks
            for them logs exactly the tags it logged before. Both are
            lower-is-better and RGB-only (no colorspace decomposition), so they
            get their own tag families ``lpips/val`` / ``dists/val`` rather than
            a key under the PSNR/SSIM scheme. ``'lpips'`` requires the
            ``[perceptual]`` extra.

        lpips_net: LPIPS backbone — ``'alex'`` (default, and what the SR
            literature usually reports), ``'vgg'`` or ``'squeeze'``. **A LPIPS
            figure is comparable only to one computed under the same backbone**,
            exactly as an SSIM figure is comparable only within one
            ``ssim_impl``. Recorded in ``hparams`` and in every artifact's
            ``sisr_meta``, so any number can be traced back. Ignored by DISTS.
    """

    crop_border: int | None = 0
    psnr_channels: list[str] = field(default_factory=lambda: ["RGB"])
    separate_psnr: bool = False
    ssim_channels: list[str] = field(default_factory=lambda: ["RGB", "Y"])
    ssim_impl: Literal["wang", "daala"] = "wang"
    perceptual_metrics: list[str] = field(default_factory=list)
    lpips_net: Literal["alex", "vgg", "squeeze"] = "alex"

    def __post_init__(self) -> None:
        """Validate all channel/metric fields at construction.

        Covers ``crop_border``, ``psnr_channels``, ``ssim_channels``,
        ``ssim_impl``, ``perceptual_metrics`` and ``lpips_net``.

        Raises:
            ValueError: If ``crop_border`` is negative (``None`` and ``0`` are
                both legitimate, and mean different things); if any entry of
                either channel field is not a
                supported colorspace or single-channel name (see
                ``_CHANNEL_SUBNAMES``), if ``ssim_impl`` is not ``'wang'`` or
                ``'daala'``, if any entry of ``perceptual_metrics`` is not a
                supported metric (see ``PERCEPTUAL_METRICS``), or if
                ``lpips_net`` is not ``'alex'``, ``'vgg'`` or ``'squeeze'``.
        """
        if self.crop_border is not None and self.crop_border < 0:
            raise ValueError(
                f"SREvalConfig.crop_border must be None or >= 0; got {self.crop_border}. "
                "`None` is the sentinel meaning 'derive the border from the model's scale', "
                "so a negative value is exactly what someone reaches for to mean 'auto' -- "
                "and it does not mean that: the slice guard treats anything <= 0 as 'no "
                "crop', so a negative border silently scores the FULL image and produces a "
                "number comparable to no published result. Fix "
                "model.eval_config.init_args.crop_border in your YAML."
            )

        valid = tuple(_CHANNEL_SUBNAMES)
        for field_name in ("psnr_channels", "ssim_channels"):
            invalid = [c for c in getattr(self, field_name) if c not in valid]
            if invalid:
                raise ValueError(
                    f"SREvalConfig.{field_name} entries must be one of "
                    f"{list(valid)}; got unsupported {invalid}. Fix "
                    f"model.eval_config.init_args.{field_name} in your YAML."
                )
        if self.ssim_impl not in ("wang", "daala"):
            raise ValueError(
                f"SREvalConfig.ssim_impl must be 'wang' or 'daala'; got "
                f"{self.ssim_impl!r}. Fix model.eval_config.init_args.ssim_impl "
                f"in your YAML."
            )
        unsupported = [m for m in self.perceptual_metrics if m not in PERCEPTUAL_METRICS]
        if unsupported:
            raise ValueError(
                f"SREvalConfig.perceptual_metrics entries must be one of "
                f"{sorted(PERCEPTUAL_METRICS)}; got unsupported {unsupported}. Fix "
                f"model.eval_config.init_args.perceptual_metrics in your YAML."
            )
        if self.lpips_net not in ("alex", "vgg", "squeeze"):
            raise ValueError(
                f"SREvalConfig.lpips_net must be 'alex', 'vgg' or 'squeeze'; got "
                f"{self.lpips_net!r}. Fix model.eval_config.init_args.lpips_net."
            )

    @property
    def psnr_keys(self) -> list[str]:
        """Ordered PSNR metric keys this config requests.

        For each entry in ``psnr_channels`` (in order), per-channel keys are
        emitted first when ``separate_psnr`` is ``True``, followed by the
        entry itself — e.g. ``psnr_channels=['RGB']`` with
        ``separate_psnr=True`` yields ``['R', 'G', 'B', 'RGB']``. Bare
        single-channel entries (e.g. ``'Y'``) have no sub-channels to expand
        (``_CHANNEL_SUBNAMES['Y'] == ()``), so they always contribute
        exactly one key regardless of ``separate_psnr``.

        This is the seam consumed by ``SRLightning`` (val metric logging /
        HParams registration) and ``BenchmarkImageLogger`` (benchmark PSNR
        key selection) — the single place the key set and its order are
        derived, so the two consumers cannot disagree.

        Returns:
            Ordered list of PSNR keys, e.g. ``['RGB', 'Y']`` or
            ``['R', 'G', 'B', 'RGB', 'Y', 'Cb', 'Cr', 'YCbCr']``.
        """
        keys: list[str] = []
        for cs in self.psnr_channels:
            if self.separate_psnr:
                keys.extend(_CHANNEL_SUBNAMES[cs])
            keys.append(cs)
        return keys

    @property
    def ssim_keys(self) -> list[str]:
        """Ordered SSIM metric keys this config requests.

        Mirrors ``psnr_keys`` over ``ssim_channels`` instead, minus the
        ``separate_psnr`` expansion step — there is no ``separate_ssim``,
        so each entry contributes exactly itself, in order.

        This is the seam consumed by ``SRLightning`` (val metric logging /
        HParams registration / checkpoint-monitor validation) and
        ``BenchmarkImageLogger`` (benchmark SSIM key selection), same as
        ``psnr_keys``.

        Returns:
            Ordered list of SSIM keys, e.g. ``['RGB', 'Y']``.
        """
        return list(self.ssim_channels)

    @property
    def perceptual_keys(self) -> list[str]:
        """Ordered perceptual metric names this config requests.

        The single derivation of the perceptual key list, mirroring
        ``psnr_keys`` / ``ssim_keys`` — consumed by ``SRLightning`` (validation
        logging, HParams registration), ``BenchmarkImageLogger`` (test-set
        scoring) and the checkpoint-monitor validator, so the three cannot
        disagree about which tags exist.

        Returns:
            e.g. ``['lpips', 'dists']``, or ``[]`` when none are requested.
        """
        return list(self.perceptual_metrics)
