"""Per-architecture training and evaluation configuration dataclasses.

Split into two classes by lifecycle:

* ``SRTrainingConfig`` controls behaviour during ``cli fit`` — per-Conv2d
  learning rates, paper-init knobs, and the example input shape used to log
  the model graph to TensorBoard.
* ``SREvalConfig`` controls validation/test metric computation —
  boundary-pixel exclusion (``crop_border``) and which colorspaces PSNR/SSIM
  are reported in.

Per-architecture defaults live in subclasses next to the model code (e.g.
``sisr.models.srcnn.SRCNNTrainingConfig``); a YAML user picks them via
``class_path`` on ``model.training_config`` / ``model.eval_config`` and
overrides individual fields with ``init_args``.

The colorspace the model trains in is no longer a string field here; it is
expressed by the choice of processor (see ``sisr.processors``).

``SRTrainingConfig.validate_against`` / ``SREvalConfig.psnr_keys`` /
``SREvalConfig.ssim_keys`` are the config-side half of the correlated-field
validation seam: fields like
``num_channels`` (model) and ``class_path`` (processor) live in sibling
objects a single dataclass ``__post_init__`` cannot see across, so the
cross-object check happens once both are constructed, orchestrated by
``SRLightning.__init__``.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch

from ..models.base import SRModel
from ..processors.base import SRProcessor

# Colorspace entries map to their sub-channel names, expanded only when
# separate_psnr=True; single-channel entries (e.g. 'Y') map to () since
# there's nothing further to decompose — this lets a bare channel name be
# requested as a first-class psnr_channels/ssim_channels entry (e.g.
# ['RGB', 'Y'] for the paper's Y-only metric without YCbCr's
# smoother-chroma-diluted aggregate).
# Doubles as the set of supported ``psnr_channels``/``ssim_channels`` entries
# (validated in `SREvalConfig.__post_init__`) — shared by both metric
# families so PSNR and SSIM cannot silently diverge on which colorspace
# names are legal.
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
        layer_lrs: Absolute per-``Conv2d`` learning rates (one entry per
            ``Conv2d`` in the model, in module-traversal order).  When
            ``None`` (default), training uses the optimizer's base ``lr``
            uniformly across all parameters.  Only valid for architectures
            where every trainable parameter lives in a ``Conv2d`` (no
            BatchNorm / PReLU); ``SRLightning.configure_optimizers``
            raises ``ValueError`` otherwise.

        example_input_shape: Shape of a single input sample *excluding* the
            batch dimension (e.g. ``(1, 33, 33)`` for a 33×33 Y-channel patch).
            When provided, ``self.example_input_array`` is set so the
            TensorBoard logger can capture the model graph, and
            :meth:`validate_against` probes the model with it.

        init_strategy: ``'paper'`` triggers a paper-faithful weight init via
            ``SRModel.reset_parameters`` in ``SRLightning``'s constructor;
            ``'default'`` (the default) skips it and uses PyTorch's defaults.
            Subclasses pin a paper-faithful default (e.g. ``SRCNNTrainingConfig``
            uses ``'paper'``).

        init_mean: Mean of the Gaussian for ``init_strategy='paper'``
            implementations. Not itself paper-derived — this shared base
            has no single paper to match; architectures with a paper init
            (e.g. ``SRCNNTrainingConfig``) override with their actual value.
            Defaults to ``0.0``.

        init_std: Std of the Gaussian for ``init_strategy='paper'``
            implementations. Not itself paper-derived — see ``init_mean``;
            e.g. ``SRCNNTrainingConfig`` overrides this to ``0.001``, the
            value its paper specifies. Defaults to ``0.01``.

        scale: The model's upscaling factor, for provenance metadata and
            cross-checking against the model's own ``scale`` hparam (see
            :meth:`validate_against`). ``None`` (the default) leaves it
            unchecked and out of exported metadata — appropriate for
            architectures like SRCNN where scale is a training-data choice,
            not a fixed property of the model itself. Deliberately not
            inferred from ``trainer.datamodule.train_dataset.scale``: that
            attribute lives outside the ``SRDataset`` contract (``PredictDataset``
            has none at all), whereas per-paper knobs already live here.

        compile_backend: Name of a ``torch._dynamo`` backend (e.g.
            ``'cudagraphs'``, ``'inductor'``) to compile the training-mode
            forward with via ``torch.compile``; ``None`` (default) trains
            eager. Only ``training_step`` is compiled — ``SRLightning``
            dispatches on ``self.training``, so validation/test/predict
            always run eager, which the widely varying benchmark image
            sizes require (CUDA-graph-style backends need static shapes).
            An unrecognized name raises immediately at ``SRLightning``
            construction (``torch._dynamo.exc.InvalidBackend``); a
            recognized name whose toolchain is missing (e.g. ``'inductor'``
            without a Triton install) only fails on first call, which
            ``SRLightning.on_fit_start`` turns into an immediate failure via
            a warm-up forward instead of an arbitrary mid-run crash.
            Measured on one RTX 5060 Laptop (SRResNet, batch 16, 24x24 LR):
            ``'cudagraphs'`` gave +4% steps/s over eager — it only captures
            the model forward, so backward and the optimizer step stay
            eager and most kernel launches in a training step are never
            captured. ``'inductor'`` fails outright there (no upstream
            Windows Triton wheel). Defaults to ``None`` so this mostly-
            unproven-on-this-project path never ships on by default.

        cuda_graph: Capture the training step's
            ``{zero_grad, forward, loss, backward}`` into a CUDA graph and
            replay it per step, instead of relaunching every kernel
            (:class:`~sisr.training.cuda_graph.CUDAGraphStep`).
            ``optimizer.step()`` stays eager, so LR schedulers, gradient
            clipping and ``global_step`` accounting are unaffected. Requires a
            CUDA device, ``precision='32-true'``, a single process, and
            ``accumulate_grad_batches=1``; ``SRLightning.on_fit_start`` refuses
            the rest rather than silently mistraining. Mutually exclusive with
            ``compile_backend`` (see :meth:`__post_init__`). Validation, test
            and predict always stay eager — their image sizes vary, and graphs
            need static shapes. Measured on one RTX 5060 Laptop (SRCNN, batch
            64, 33x33 Y patches, 60 W cap): **2.81x steps/s**, 6.21 -> 2.21
            ms/step, bit-identical losses. The win is proportional to how
            launch-bound the architecture is, so it is far smaller for
            SRResNet, whose GPU floor is real compute — hence opt-in, per
            config, defaulting off.
    """

    layer_lrs: list[float] | None = None
    example_input_shape: tuple[int, ...] | None = None
    init_strategy: Literal["default", "paper"] = "default"
    init_mean: float = 0.0
    init_std: float = 0.01
    scale: int | None = None
    compile_backend: str | None = None
    cuda_graph: bool = False

    def __post_init__(self) -> None:
        """Reject field combinations that cannot both be honoured.

        Raises:
            ValueError: If ``cuda_graph`` and ``compile_backend`` are both set.
        """
        if self.cuda_graph and self.compile_backend is not None:
            raise ValueError(
                f"training_config.cuda_graph=True is incompatible with "
                f"compile_backend={self.compile_backend!r}: both take over the "
                f"training-mode forward, and each needs its own warm-up before it is "
                f"usable, so layering them means capturing a graph around a partially "
                f"warmed compiled callable. Pick one — set compile_backend=null to keep "
                f"cuda_graph, or cuda_graph=false to keep compile_backend."
            )

    def validate_against(self, model: SRModel, processor: SRProcessor) -> None:
        """Validate this config against the model/processor it will pair with.

        Universal, architecture-agnostic checks:

        - When ``self.scale`` is set and ``model.hparams`` declares a
          ``'scale'`` entry, the two must agree. Either side being absent
          (``self.scale is None``, or the model has no ``'scale'`` hparam —
          e.g. SRCNN) skips the check silently rather than guessing.
        - When ``example_input_shape`` is set, its channel dimension must
          equal ``processor.model_channels``, and a ``torch.no_grad()``
          forward pass of the real ``model`` on a zero tensor of that shape
          must succeed. The probe exercises the actual ``nn.Module`` rather
          than a separate description of it, so it cannot go stale as the
          architecture evolves. This half is a no-op when
          ``example_input_shape`` is unset — it is optional (TensorBoard
          graph / FLOPs reporting only).

        Subclasses (e.g. ``SRCNNTrainingConfig``) override this to add
        architecture-specific correlation checks — e.g. ``num_channels`` vs
        ``processor.model_channels`` — that exist purely to raise an
        actionable message *before* a mismatched pairing would otherwise
        surface as a raw shape-mismatch error from this probe. They call
        ``super().validate_against(model, processor)`` to keep the universal
        checks.

        Args:
            model: The constructed :class:`~sisr.models.base.SRModel` this
                config will train/evaluate.
            processor: The :class:`~sisr.processors.base.SRProcessor` paired
                with ``model`` for this run.

        Raises:
            ValueError: If ``self.scale`` is set and disagrees with the
                model's own ``scale`` hparam, or if ``example_input_shape``
                is set and its channel dimension doesn't match
                ``processor.model_channels``.
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
        crop_border: Number of border pixels to exclude on each edge before
            computing PSNR / SSIM.  Standard SR-evaluation convention is to
            crop the outer ``scale`` pixels (e.g. ``crop_border=3`` for x3).

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
    """

    crop_border: int = 0
    psnr_channels: list[str] = field(default_factory=lambda: ["RGB"])
    separate_psnr: bool = False
    ssim_channels: list[str] = field(default_factory=lambda: ["RGB", "Y"])

    def __post_init__(self) -> None:
        """Validate ``psnr_channels`` and ``ssim_channels`` at construction.

        Raises:
            ValueError: If any entry of either field is not a supported
                colorspace or single-channel name (see ``_CHANNEL_SUBNAMES``).
        """
        valid = tuple(_CHANNEL_SUBNAMES)
        for field_name in ("psnr_channels", "ssim_channels"):
            invalid = [c for c in getattr(self, field_name) if c not in valid]
            if invalid:
                raise ValueError(
                    f"SREvalConfig.{field_name} entries must be one of "
                    f"{list(valid)}; got unsupported {invalid}. Fix "
                    f"model.eval_config.init_args.{field_name} in your YAML."
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
