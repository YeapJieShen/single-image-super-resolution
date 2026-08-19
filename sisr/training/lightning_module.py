"""Generic Lightning wrapper for SR models — :class:`SRLightning`.

The Lightning module is architecture-agnostic: per-paper behavior lives
in :class:`~sisr.training.config.SRTrainingConfig` /
:class:`~sisr.training.config.SREvalConfig` subclasses (e.g.
:class:`~sisr.models.srcnn.SRCNNTrainingConfig`), not in the module
itself. Optimizer / LR scheduler are wired in from top-level YAML keys by
:class:`~sisr.cli.SRLightningCLI`.
"""

import dataclasses
from typing import Any

import lightning
import torch
import torchvision
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ..losses import SRLoss
from ..models.base import SRModel
from ..processors import SRProcessor
from .config import SREvalConfig, SRTrainingConfig
from .cuda_graph import CUDAGraphStep
from .metadata import build_metadata
from .probe import probe_pair
from .scoring import SRScorer, metric_tag


class SRLightning(lightning.LightningModule):
    """Generic Lightning wrapper for single-image super-resolution models.

    Composes an :class:`~sisr.models.base.SRModel` with an
    :class:`~sisr.processors.SRProcessor`. The model is a pure tensor function;
    the processor handles per-batch colorspace conversion between the dataset's
    RGB format and the model's input/output colorspace. Both are wired
    independently in YAML via ``class_path``.

    Args:
        model: An initialised :class:`SRModel` subclass (e.g. :class:`SRCNN`,
            :class:`SRResNet`). Required.
        processor: An :class:`SRProcessor` subclass (e.g.
            :class:`~sisr.processors.RGBProcessor`,
            :class:`~sisr.processors.YChannelProcessor`,
            :class:`~sisr.processors.YCbCrProcessor`). Required.
        training_config: Per-architecture training settings (layer_lrs,
            example_input_shape, init_*). Defaults to base
            :class:`SRTrainingConfig` (uniform LR, no paper init).
        eval_config: Per-architecture evaluation settings (crop_border,
            psnr_channels, separate_psnr, ssim_channels). Defaults to base
            :class:`SREvalConfig`.
        criterion: Loss instance. Defaults to :class:`torch.nn.MSELoss`. An
            :class:`~sisr.losses.SRLoss` additionally gets ``bind(processor)``
            called once here, so it can adapt to the model's output space.
        optimizer: ``OptimizerCallable`` populated from top-level YAML
            ``optimizer:``. Defaults to :class:`torch.optim.Adam`.
        lr_scheduler: ``LRSchedulerCallable`` or ``None``. Defaults to ``None``.

    Raises:
        TypeError: If ``model`` is not an :class:`SRModel` subclass, or
            if ``processor`` is not an :class:`SRProcessor` subclass.
    """

    def __init__(
        self,
        model: SRModel,
        processor: SRProcessor,
        training_config: SRTrainingConfig | None = None,
        eval_config: SREvalConfig | None = None,
        criterion: torch.nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ):
        super().__init__()
        if not isinstance(model, SRModel):
            raise TypeError(
                f"SRLightning requires an SRModel subclass; got "
                f"{type(model).__name__}. Update your YAML to point at a "
                f"class under sisr.models.* that inherits from SRModel."
            )
        if not isinstance(processor, SRProcessor):
            raise TypeError(
                f"SRLightning requires an SRProcessor subclass; got "
                f"{type(processor).__name__}. Update your YAML to point "
                f"at a class under sisr.processors.* that inherits from "
                f"SRProcessor."
            )

        self.model = model
        self.processor = processor
        self.training_config = training_config or SRTrainingConfig()
        self.eval_config = eval_config or SREvalConfig()
        # The scoring path's one owner: crop, colorspace split, both
        # reductions and the tag grammar. BenchmarkImageLogger uses this
        # same object, so the two paths cannot disagree.
        self.scorer = SRScorer(self.eval_config)
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.training_config.validate_against(self.model, self.processor)

        if isinstance(self.criterion, SRLoss):
            self.criterion.bind(self.processor)

        if self.training_config.init_strategy == "paper":
            model.reset_parameters(
                mean=self.training_config.init_mean,
                std=self.training_config.init_std,
            )

        backend = self.training_config.compile_backend
        compiled = torch.compile(self.model, backend=backend) if backend is not None else None
        # torch.compile() returns an OptimizedModule, itself an nn.Module — a
        # plain `self._compiled = compiled` would register it as a submodule
        # (nn.Module.__setattr__ registers any nn.Module value regardless of
        # the leading underscore), duplicating every parameter under
        # "_compiled._orig_mod.*" in state_dict() and breaking strict-mode
        # checkpoint loading against a module built with compilation off.
        # object.__setattr__ bypasses that registration; self.model and
        # compiled._orig_mod stay the same object, so there is exactly one
        # set of weights and it still moves with .to() through self.model's
        # normal registration.
        object.__setattr__(self, "_compiled", compiled)

        self._cuda_graph: CUDAGraphStep | None = None
        # Per-step, not per-run: on an eager fallback step Lightning must still
        # run backward and zero_grad. Gating those on "a graph exists" instead
        # makes the epoch's partial last batch step on the previous replay's
        # gradients — a silent duplicated update, not a crash.
        self._graphed_step = False

        # Save exactly this plain dict — bypasses save_hyperparameters' frame/given_hparams
        # introspection, so the checkpoint's `hyper_parameters` is identical whether this
        # module is built directly or via SRLightningCLI. dataclasses.asdict (not the live
        # training_config/eval_config objects) keeps it weights_only=True loadable and free
        # of the two dataclass types, which torch.load's safe-globals allowlist doesn't know.
        # This dict is what LightningCLI._parse_ckpt_path reads back and re-parses as CLI
        # options (model.<key>) on `--ckpt_path` reload — it must stay nested, not flattened
        # with _flatten_hparams' '/' separator, which jsonargparse can't parse as an option name.
        self.save_hyperparameters(
            {
                "training_config": dataclasses.asdict(self.training_config),
                "eval_config": dataclasses.asdict(self.eval_config),
            }
        )

        # TensorBoard-only view: flattened (see _flatten_hparams) and enriched with the
        # model's own hparams + processor/criterion identity for HParams columns. Kept off
        # self.hparams (and so off the checkpoint) — on_train_start logs this instead.
        self._tb_hparams = self._flatten_hparams(
            {
                **self.hparams,
                "model": model.hparams,
                "processor": type(processor).__name__,
                "criterion": self.criterion_description,
            }
        )

        if self.training_config.example_input_shape is not None:
            self.example_input_array = torch.zeros(1, *self.training_config.example_input_shape)

    @property
    def criterion_description(self) -> str:
        """One-line human-readable identity of the criterion.

        The single derivation point for the TensorBoard HParams column and
        for checkpoint/export provenance metadata, so the two cannot drift —
        mirroring how ``SREvalConfig.psnr_keys`` is the one derivation for
        its consumers.
        """
        return (
            self.criterion.describe()
            if isinstance(self.criterion, SRLoss)
            else type(self.criterion).__name__
        )

    def setup(self, stage: str | None = None) -> None:
        """Probe one real sample against ``model.input_contract`` / ``example_input_shape``.

        Lightning always runs ``DataModule.setup(stage)`` before
        ``LightningModule.setup(stage)`` (``Trainer._call_setup_hook``), so
        ``trainer.datamodule``'s stage-relevant datasets are already
        instantiated by the time this runs. Cross-wiring a model and dataset
        with mismatched ``input_contract`` (e.g. SRResNet's ``native_lr``
        trained on SRCNN's ``pre_upsampled`` dataset, or the mirror) produces
        no error downstream otherwise — :meth:`_forward_sr`'s ``center_crop``
        silently zero-pads instead, and every loss/PSNR number that follows is
        meaningless. Checking one real sample here is cheap and fires before
        the first step.

        Picks the first dataset ``trainer.datamodule`` has actually
        instantiated for this stage — train, else the primary val set, else
        the first test set — so the check runs for ``fit`` as well as a
        bare ``validate``/``test`` invocation, not only ``fit``.
        ``PredictDataset`` (bare LR tensor, no HR) is never in that list, so
        a predict-only datamodule has nothing to check. Separately, when a
        train dataset exists and ``training_config.example_input_shape`` is
        set, its spatial dims are checked against the real train LR patch —
        train-only, since validation/test images vary in size.

        No-op when no ``Trainer`` is attached (direct/unit-test construction),
        when ``trainer.datamodule`` has nothing instantiated yet for this
        stage, or when ``trainer.datamodule`` is some foreign object that
        doesn't expose the ``train_dataset``/``val_dataset``/``test_datasets``
        read accessors (degrades to a skip rather than ``AttributeError``).

        The sampling discipline this needs — RNG transparency and pickle-clone
        reads, both load-bearing and neither obvious at the call site — belongs
        to :mod:`sisr.training.probe` and is described there. This method owns
        only the *checks*; :func:`~sisr.training.probe.probe_pair` owns getting
        a sample without side effects, and the ``with`` block is what
        guarantees everything below runs inside its guard.

        Args:
            stage: Lightning trainer stage — ``'fit'``, ``'validate'``,
                ``'test'``, or ``'predict'``.

        Raises:
            ValueError: If the sampled LR/HR pair disagrees with
                ``self.model.input_contract``, or (train stage only) if
                ``training_config.example_input_shape``'s spatial dims don't
                match the real train LR sample.
        """
        dm = self._trainer.datamodule if self._trainer is not None else None
        if dm is None:
            return

        with probe_pair(dm) as probe:
            if probe.sample is not None:
                s = probe.sample
                self._check_input_contract(s.lr, s.hr, s.source, s.dataset)
                self._extra_probe(s.lr, s.hr, s.source)

            if self.training_config.example_input_shape is not None:
                train = probe.train_lr()
                if train is not None:
                    train_dataset, train_lr = train
                    self._check_example_input_shape(train_lr, train_dataset)

    def _check_input_contract(
        self, lr: torch.Tensor, hr: torch.Tensor, source: str, dataset: Any
    ) -> None:
        """Raise if the sampled ``(lr, hr)`` pair disagrees with ``model.input_contract``.

        Args:
            lr: LR sample tensor, shape ``(C, H, W)``.
            hr: HR sample tensor, shape ``(C, H, W)``.
            source: Config path of the dataset the sample came from (e.g.
                ``'train_dataset'``), for the error message.
            dataset: The dataset instance, for its class name in the error.

        Raises:
            ValueError: See :meth:`setup`, or if ``model.input_contract`` is
                neither ``'pre_upsampled'`` nor ``'native_lr'``.
        """
        contract = self.model.input_contract
        lr_hw, hr_hw = tuple(lr.shape[-2:]), tuple(hr.shape[-2:])
        ds_name = type(dataset).__name__

        if contract == "pre_upsampled":
            if lr_hw == hr_hw:
                return
            raise ValueError(
                f"{type(self.model).__name__}.input_contract='pre_upsampled' requires "
                f"LR and HR to share spatial size, but data.{source} ({ds_name}) served "
                f"lr {lr_hw} != hr {hr_hw}. A native-LR dataset (e.g. "
                f"sisr.datasets.srresnet) paired with a pre-upsampled model silently "
                f"zero-pads in SRLightning._forward_sr instead of raising — point "
                f"data.{source} at a pre-upsampled dataset (e.g. sisr.datasets.srcnn), "
                f"or switch to a native_lr model."
            )

        if contract != "native_lr":
            raise ValueError(
                f"{type(self.model).__name__}.input_contract={contract!r} is not a "
                f"recognised contract — expected 'pre_upsampled' or 'native_lr'. Fix "
                f"{type(self.model).__name__}.input_contract."
            )

        # Fall back to the model's own 'scale' hparam when training_config.scale
        # is unset (e.g. a bare SRTrainingConfig()) — only skip the check when
        # BOTH are absent. Falling back keeps this from silently no-opping for
        # exactly the native_lr construction several existing tests use.
        scale = self.training_config.scale
        if scale is None:
            scale = self.model.hparams.get("scale")
        if scale is None:
            return
        expected_hw = (lr_hw[0] * scale, lr_hw[1] * scale)
        if hr_hw == expected_hw:
            return
        raise ValueError(
            f"{type(self.model).__name__}.input_contract='native_lr' requires "
            f"hr.shape[-2:] == lr.shape[-2:] * training_config.scale ({scale}), but "
            f"data.{source} ({ds_name}) served lr {lr_hw}, hr {hr_hw} (expected "
            f"{expected_hw}). A pre-upsampled dataset (e.g. sisr.datasets.srcnn) "
            f"paired with a native-LR model silently zero-pads in "
            f"SRLightning._forward_sr instead of raising — point data.{source} at a "
            f"native-LR dataset (e.g. sisr.datasets.srresnet), or fix "
            f"training_config.scale."
        )

    def _extra_probe(self, lr: torch.Tensor, hr: torch.Tensor, source: str) -> None:
        """Subclass hook: additional checks against a real ``(lr, hr)`` sample.

        No-op by default. Called from inside
        :func:`~sisr.training.probe.probe_pair`'s guarded block, so a subclass
        gets RNG transparency and pickle-clone sampling without asking for them
        — and cannot opt out of them either. A subclass that needs its *own*
        sample should use ``probe_pair`` directly rather than reading a dataset,
        for the reasons that module documents.

        Args:
            lr: LR sample, ``(C, H, W)``.
            hr: HR sample, ``(C, H, W)``.
            source: Config path the sample came from, for error messages.
        """

    def _check_example_input_shape(self, lr: torch.Tensor, train_dataset: Any) -> None:
        """Raise if ``example_input_shape``'s H/W disagrees with the real train LR patch.

        Args:
            lr: Real LR sample from the train dataset, shape ``(C, H, W)``.
            train_dataset: The train dataset instance, for its class name in
                the error.

        Raises:
            ValueError: If ``example_input_shape[1:] != lr.shape[-2:]``.
        """
        expected_hw = tuple(self.training_config.example_input_shape[1:])
        actual_hw = tuple(lr.shape[-2:])
        if expected_hw == actual_hw:
            return
        raise ValueError(
            f"training_config.example_input_shape has spatial dims {expected_hw}, but "
            f"data.train_dataset ({type(train_dataset).__name__}) serves LR patches of "
            f"{actual_hw}. example_input_shape drives the compile warm-up shape and "
            f"FLOPs reporting — set it to match the true train patch (hr_crop_size // "
            f"scale for SRResNet-style datasets, subimg_size for SRCNN-style)."
        )

    @staticmethod
    def _flatten_hparams(hparams: dict[str, Any], sep: str = "/") -> dict:
        """Recursively flatten nested hparams dicts/lists for clean TensorBoard HParams columns.

        None values are dropped (they add noise without aiding comparison).
        Class objects are reduced to their ``__name__``.
        """
        result = {}

        def _recurse(obj, prefix: str) -> None:
            if obj is None:
                return
            if isinstance(obj, type):
                result[prefix] = obj.__name__
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _recurse(v, f"{prefix}{sep}{k}")
            elif isinstance(obj, list | tuple):
                for i, v in enumerate(obj):
                    _recurse(v, f"{prefix}{sep}{i}")
            elif isinstance(obj, bool | int | float | str):
                result[prefix] = obj
            else:
                result[prefix] = str(obj)

        for k, v in hparams.items():
            _recurse(v, k)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapped SR model on ``x`` and return its raw output.

        Pure inference path — no colorspace conversion, no metrics, no
        cropping. Used by TensorBoard graph logging (tracing
        ``example_input_array``) and direct ``module(x)`` calls — NOT the
        ``trainer.predict`` path, which :meth:`predict_step` overrides,
        calling :meth:`_forward_lr` (extract → model → reconstruct)
        instead. Training and validation likewise bypass this method,
        going through :meth:`_step`, which adds the same colorspace
        pipeline.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``, already in the
                model's own IO colorspace (e.g. Y for SRCNN, RGB for
                SRResNet).

        Returns:
            SR tensor as produced by the wrapped model. SRResNet upscales
            spatially to ``(B, C, H*scale, W*scale)``; SRCNN preserves
            H/W only when ``padding='same'`` — the default ``'valid'``
            (or an explicit int) shrinks each dim by
            ``kernel_size - 1 - 2*padding`` per conv layer instead (see
            :meth:`~sisr.models.srcnn.model.SRCNN.forward`).
        """
        return self.model(x)

    def _forward_lr(
        self, lr_img: torch.Tensor, need_sr_rgb: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """LR-only core of the forward pipeline: extract → model → (reconstruct).

        Factored out of :meth:`_forward_sr` so :meth:`predict_step` can share
        the exact colorspace pipeline instead of forking it — HR is only ever
        needed for the center-crop :meth:`_forward_sr` adds on top, which has
        no meaning without an HR reference at inference time.

        Dispatches to the ``torch.compile``d model iff ``self.training`` and
        ``training_config.compile_backend`` is set; ``self.training`` is
        exactly right here since Lightning maintains it and training uses
        fixed patch shapes while validation/predict see widely varying
        image sizes (see ``training_config.compile_backend``'s docstring).

        Args:
            lr_img: LR batch, RGB ``float32`` in ``[0, 1]``, shape
                ``(B, 3, H, W)``.
            need_sr_rgb: Whether to run ``processor.reconstruct`` at all.
                ``training_step`` (via :meth:`_step`) is the only caller that
                passes ``False``: it discards ``sr_rgb`` entirely
                (``loss, *_ = self._step(...)``), yet reconstruct still costs
                real per-step time (measured: ~11.5% of a data-free SRCNN
                step for ``YChannelProcessor``'s bicubic Cb/Cr interpolate;
                ~0 for SRResNet's identity/elementwise processors). Every
                other caller — validation, test, predict, and direct calls
                like these from tests — keeps the default ``True`` and gets
                exactly the reconstruction this project has always produced.

        Returns:
            ``(sr_model_out, sr_rgb)`` — the raw model output in the model IO
            colorspace, and the reconstructed SR RGB clamped to ``[0, 1]``;
            ``sr_rgb`` is ``None`` iff ``need_sr_rgb`` is ``False``.
        """
        model_input = self.processor.extract(lr_img)
        model_fn = self._compiled if self.training and self._compiled is not None else self.model
        sr_model_out = model_fn(model_input)
        if not need_sr_rgb:
            return sr_model_out, None
        sr_rgb = self.processor.reconstruct(sr_model_out, lr_img)
        # Clamp display-space output only, here, once — every reconstruct()
        # consumer (_forward_sr's callers, predict_step) reads through this
        # one line, so PSNR/SSIM never diverge from what an 8-bit image would
        # score. sr_model_out (the loss target) is untouched: clamping
        # it would kill gradients on saturated pixels. Idempotent where
        # reconstruct() already clamps (YCbCrProcessor/YChannelProcessor's
        # ycbcr_to_rgb) — don't move this into forward() or a processor, or
        # the bound would depend on training colorspace again.
        sr_rgb = sr_rgb.clamp(0.0, 1.0)
        return sr_model_out, sr_rgb

    def _forward_sr(
        self, lr_img: torch.Tensor, hr_img: torch.Tensor, need_sr_rgb: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Canonical SR forward: extract → model → (reconstruct) → crop HR.

        The single source of truth for the forward pipeline. Shared by
        :meth:`_step` (which also needs the model-space output for the loss),
        :meth:`predict_rgb` (the public scoring seam), and — via
        :meth:`_forward_lr` — :meth:`predict_step` (the HR-free inference
        seam), so the training, benchmark-logging, and prediction paths
        cannot silently diverge.

        Args:
            lr_img: LR batch, RGB ``float32`` in ``[0, 1]``, shape
                ``(B, 3, H, W)``.
            hr_img: HR batch, RGB ``float32`` in ``[0, 1]``.
            need_sr_rgb: Forwarded to :meth:`_forward_lr` — see there. The
                HR crop uses ``sr_model_out.shape[-2:]`` regardless, since
                every shipped ``SRProcessor.reconstruct`` preserves H/W, so
                it always agrees with what ``sr_rgb.shape[-2:]`` would be.

        Returns:
            ``(sr_model_out, sr_rgb, hr_cropped)`` — the raw (unclamped)
            model output in the model IO colorspace, the reconstructed SR RGB
            clamped to ``[0, 1]`` (``None`` iff ``need_sr_rgb`` is ``False``),
            and HR center-cropped to the SR spatial size.

        Raises:
            ValueError: If ``hr_img`` is spatially smaller than the model
                output in either dimension — ``center_crop`` would zero-pad
                instead of cropping, silently corrupting the loss/metrics
                that follow. :meth:`setup` catches the common cause (a
                model/dataset ``input_contract`` mismatch) earlier and
                louder; this is the last-resort guard for callers that
                bypass it (e.g. direct ``_forward_sr``/``_step`` calls in
                tests, or a datamodule with no ``trainer`` attached).
        """
        sr_model_out, sr_rgb = self._forward_lr(lr_img, need_sr_rgb=need_sr_rgb)
        hr_hw, sr_hw = hr_img.shape[-2:], sr_model_out.shape[-2:]
        if hr_hw[0] < sr_hw[0] or hr_hw[1] < sr_hw[1]:
            raise ValueError(
                f"hr_img spatial size {tuple(hr_hw)} is smaller than the model output "
                f"{tuple(sr_hw)} — torchvision.transforms.functional.center_crop "
                f"would zero-pad instead of cropping, silently corrupting the loss "
                f"and every metric downstream. This usually means the dataset's "
                f"LR/HR pairing doesn't match {type(self.model).__name__}."
                f"input_contract={self.model.input_contract!r} — check "
                f"data.train_dataset/val_dataset/test_datasets in your config."
            )
        hr_cropped = torchvision.transforms.functional.center_crop(hr_img, sr_hw)
        return sr_model_out, sr_rgb, hr_cropped

    def predict_rgb(
        self, lr_img: torch.Tensor, hr_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the SR forward and return spatially-aligned SR / HR RGB tensors.

        Public scoring seam consumed by the training path (via :meth:`_step`)
        and by :class:`~sisr.training.callbacks.BenchmarkImageLogger`, so
        neither re-implements the ``extract → model → reconstruct →
        center_crop`` pipeline nor reaches into ``.model`` / ``.processor``.
        Gradient tracking follows the caller's context — wrap in
        ``torch.no_grad()`` for pure inference.

        Args:
            lr_img: LR batch, RGB ``float32`` in ``[0, 1]``, shape
                ``(B, 3, H, W)``.
            hr_img: HR batch, RGB ``float32`` in ``[0, 1]``.

        Returns:
            ``(sr_rgb, hr_cropped)`` — SR RGB clamped to ``[0, 1]`` and HR
            center-cropped to the SR spatial size, both RGB ``float32``.
        """
        _, sr_rgb, hr_cropped = self._forward_sr(lr_img, hr_img)
        return sr_rgb, hr_cropped

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor], need_sr_rgb: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Shared forward + loss for training and validation steps.

        The processor handles all colorspace conversion; loss is computed in
        the model's *output* space (against HR mapped into it via
        ``processor.extract_target``, which defaults to ``processor.extract``
        and differs only where a model's input and output ranges differ).
        Metrics downstream consume ``sr_rgb`` / ``hr_cropped`` (both full
        RGB, spatially aligned) — unavailable (``None``) when the caller
        passed ``need_sr_rgb=False``, which is only correct for callers (like
        ``training_step``) that never look at ``sr_rgb``.

        Args:
            batch: ``(lr_img, hr_img)`` tuple from a loader. Both RGB,
                ``float32`` in ``[0, 1]``.
            need_sr_rgb: Forwarded to :meth:`_forward_sr` — see there.
                ``training_step`` passes ``False``.

        Returns:
            ``(loss, lr_img, hr_img, sr_rgb, hr_cropped)``. ``loss`` is a
            scalar tensor computed on the unclamped model output; ``sr_rgb``
            (clamped to ``[0, 1]``, or ``None`` iff ``need_sr_rgb`` is
            ``False``) and ``hr_cropped`` are RGB tensors with matching
            spatial size.
        """
        lr_img, hr_img = batch

        sr_model_out, sr_rgb, hr_cropped = self._forward_sr(lr_img, hr_img, need_sr_rgb=need_sr_rgb)
        hr_for_loss = self.processor.extract_target(hr_cropped)
        loss = self.criterion(sr_model_out, hr_for_loss)

        return loss, lr_img, hr_img, sr_rgb, hr_cropped

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute training loss for one batch and log it.

        Delegates the forward + colorspace + loss pipeline to :meth:`_step`
        and logs ``loss/train`` on every step for the progress bar, plus
        ``loss/train/{term}`` for each term of a composite criterion.
        ``need_sr_rgb=False``: this method never looks at ``sr_rgb``, so
        skips ``processor.reconstruct`` — real per-step time (see
        :meth:`_forward_lr`) for a value that would only be discarded.

        Args:
            batch: ``(lr_img, hr_img)`` tuple as produced by the
                :class:`SRDataModule` train loader. Both are ``float32`` in
                ``[0, 1]``.
            batch_idx: Index of the batch within the current epoch.

        Returns:
            Scalar loss tensor for the optimizer.
        """
        loss = self._graph_step(batch) if self.training_config.cuda_graph else None
        if loss is None:
            loss, *_ = self._step(batch, need_sr_rgb=False)
        self.log("loss/train", loss, prog_bar=True, on_step=True)
        self._log_loss_terms("train")
        return loss

    def _log_loss_terms(self, stage: str) -> None:
        """Log a composite criterion's per-term contributions, if it has any.

        Reads ``last_terms`` structurally rather than by type, so any
        criterion exposing that mapping participates. Empty (and so a no-op)
        for every scalar loss.

        Args:
            stage: ``"train"`` or ``"val"`` — the middle tag segment.
        """
        terms = getattr(self.criterion, "last_terms", None)
        if not terms:
            return
        on_step = stage == "train"
        for name, value in terms.items():
            self.log(
                f"loss/{stage}/{name}",
                value,
                on_step=on_step,
                on_epoch=not on_step,
                add_dataloader_idx=False,
            )

    def _graph_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | None:
        """Run the captured training step, building the graph on first use.

        Args:
            batch: ``(lr_img, hr_img)`` tuple, already on the device.

        Returns:
            The replayed step's loss, or ``None`` when the batch's shapes don't
            match the captured ones (the epoch's partial last batch) and the
            caller must run an eager step instead.
        """
        if self._cuda_graph is None:
            optimizer = self.optimizers()
            self._cuda_graph = CUDAGraphStep(
                lambda b: self._step(b, need_sr_rgb=False)[0],
                self,
                getattr(optimizer, "optimizer", optimizer),
            )
            reason = self._unsafe_capture_reason()
            if reason is not None:
                self._cuda_graph.disable(reason)
        loss = self._cuda_graph.run(batch)
        self._graphed_step = loss is not None
        return loss

    def _unsafe_capture_reason(self) -> str | None:
        """Why capture must not even be attempted on this run, or ``None``.

        A ``DataLoader`` with ``num_workers > 0`` **and** ``pin_memory=True``
        runs its pinning in a separate thread of *this* process, and that thread
        calls ``cudaHostAlloc`` whenever its host-block cache needs a new size.
        ``cudaHostAlloc`` implicitly synchronizes the device, which invalidates
        any capture open at that moment. Measured on this project: 1 crash in 6
        fits at ``num_workers=16, pin_memory=True`` versus 0 in 8 with
        ``pin_memory=False``, all else equal.

        Retrying does not make this safe. The invalidation leaves a sticky CUDA
        error that the *next* CUDA call in the process consumes, and when that
        call belongs to the pin-memory thread the ``DataLoader`` dies
        ("Caught AcceleratorError in pin memory thread") no matter what this
        process's main thread does about its own capture. Prevention is the only
        option, so this is a warn-and-run-eager condition rather than one of
        :meth:`_check_cuda_graph_prerequisites`'s hard refusals: those catch
        silent mistraining, this only costs speed.

        ``num_workers=0`` — what both shipped templates use — pins inline on the
        main thread, so there is no second thread to race and ``pin_memory``
        stays worth having there.

        Returns:
            An operator-facing reason, or ``None`` when capture is safe to try.
        """
        loader = getattr(self.trainer, "train_dataloader", None)
        workers = getattr(loader, "num_workers", 0) or 0
        if workers > 0 and getattr(loader, "pin_memory", False):
            return (
                f"the train DataLoader combines num_workers={workers} with "
                f"pin_memory=True, whose pin-memory thread calls cudaHostAlloc and can "
                f"invalidate a capture mid-flight, killing the run. Training continues "
                f"eagerly. Set pin_memory: false to graph at this worker count, or "
                f"num_workers: 0 (both shipped templates do) to keep pin_memory — it "
                f"pins on the main thread there, with nothing to race."
            )
        return None

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Backpropagate ``loss``, unless the captured graph already did.

        On an eager fallback step this writes into the same ``.grad`` tensors the
        captured graph holds addresses for, which is only safe while
        ``create_graph`` stays ``False`` — a double-backward pass makes ``.grad``
        a graph-tracked tensor and can rebind it, invalidating every later replay.
        Nothing in this project passes ``create_graph``.

        Args:
            loss: Scalar loss returned by :meth:`training_step`.
            *args: Forwarded to Lightning's implementation.
            **kwargs: Forwarded to Lightning's implementation.
        """
        if self._graphed_step:
            return
        super().backward(loss, *args, **kwargs)

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer
    ) -> None:
        """Zero gradients, unless the captured graph owns them.

        Lightning's closure order is ``training_step -> zero_grad -> backward``,
        so on a graphed step this must be a no-op: the replay has already
        zeroed, filled and finished with the gradients, and zeroing again here
        would hand the optimizer nothing but zeros. On an eager fallback step it
        must run — but with ``set_to_none=False`` whenever a graph is live,
        because the graph writes into those exact ``.grad`` tensors and freeing
        them invalidates every later replay. With no live graph (capture
        disabled after repeated failures) the ordinary ``set_to_none=True`` is
        both safe and marginally faster.

        Args:
            epoch: Current epoch index.
            batch_idx: Current batch index.
            optimizer: The optimizer whose gradients to zero.
        """
        if not self.training_config.cuda_graph:
            super().optimizer_zero_grad(epoch, batch_idx, optimizer)
        elif not self._graphed_step:
            live = self._cuda_graph is not None and self._cuda_graph.captured
            optimizer.zero_grad(set_to_none=not live)

    def on_validation_model_zero_grad(self) -> None:
        """Skip Lightning's pre-validation gradient release while a graph is live.

        Before each mid-training validation run Lightning frees gradient memory
        with ``zero_grad(set_to_none=True)``. The replay fills the exact
        ``.grad`` tensors allocated at capture and never re-binds the
        parameters' ``.grad`` attributes, so one release severs the link
        permanently: every later ``optimizer.step()`` sees ``None`` grads and
        skips every parameter — training silently freezes at the first
        validation while the logged loss keeps moving with the incoming
        batches. There is also nothing to free: the tensors live in the
        graph's private memory pool for the rest of the run either way. Eager
        runs keep Lightning's release.
        """
        live = self._cuda_graph is not None and self._cuda_graph.captured
        if not live:
            super().on_validation_model_zero_grad()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute and log validation metrics for the *primary* val loader (idx 0).

        Benchmark / test loaders (idx >= 1) are handled by
        :class:`~sisr.training.callbacks.BenchmarkImageLogger`.

        Args:
            batch: ``(lr_img, hr_img)`` tuple from the val loader.
            batch_idx: Index of the batch within the current val epoch.
            dataloader_idx: Index of the loader within the list returned
                by :meth:`SRDataModule.val_dataloader`. ``0`` is the
                primary val set; anything else is skipped.
        """
        if dataloader_idx != 0:
            return

        loss, lr_img, hr_img, sr, hr_cropped = self._step(batch)

        scores = self.scorer.score(sr, hr_cropped)

        # add_dataloader_idx=False keeps metric names clean — needed because the
        # primary val loader is at idx 0 of a list that also contains test loaders.
        self.log("loss/val", loss, prog_bar=True, on_step=False, add_dataloader_idx=False)
        self._log_loss_terms("val")
        # prog_bar is a display choice, so it stays with the caller rather than
        # moving into the scorer — the scorer decides values, not presentation.
        prog_bar_tags = {
            metric_tag("psnr", "val", self.eval_config.psnr_channels[0]),
            metric_tag("ssim", "val", self.eval_config.ssim_channels[0]),
        }
        for tag, value in scores.tagged("val").items():
            self.log(
                tag,
                value,
                prog_bar=(tag in prog_bar_tags),
                on_step=False,
                add_dataloader_idx=False,
            )

    def on_fit_start(self) -> None:
        """Check the CUDA-graph prerequisites, then warm up the compiled training path.

        A backend name ``torch._dynamo.list_backends()`` recognizes can
        still lack its toolchain (e.g. ``'inductor'`` without Triton), which
        only surfaces on the *first* call to the compiled model, not at
        ``torch.compile()`` construction time. Running one forward here
        turns that into an immediate failure at the start of ``fit``
        instead of an arbitrary point mid-run — ``num_sanity_val_steps``
        running first would not catch it, since validation always runs the
        eager path (see :meth:`_forward_lr`).

        The warm-up is a no-op when compilation is off, or when
        ``training_config.example_input_shape`` is unset — there is then no
        input shape to probe with (both shipped templates set it).

        Any graph captured by a previous ``fit`` on this module is dropped here,
        unconditionally. ``Strategy.teardown`` moves the module — and its
        ``.grad`` tensors — back to CPU at the end of a fit, which frees the
        device blocks the graph baked addresses for; replaying it in a second
        ``fit`` would write into freed memory. Re-capturing is cheap (three
        warm-up iterations) and happens on the next graphed step.

        Raises:
            RuntimeError: Via :meth:`_check_cuda_graph_prerequisites` when
                ``training_config.cuda_graph`` is set on a run that cannot
                honour it.
        """
        self._cuda_graph = None
        self._graphed_step = False
        if self.training_config.cuda_graph:
            self._check_cuda_graph_prerequisites()
        if self._compiled is None or self.training_config.example_input_shape is None:
            return
        dummy = torch.zeros(1, *self.training_config.example_input_shape, device=self.device)
        with torch.no_grad():
            self._compiled(dummy)

    def on_train_epoch_start(self) -> None:
        """Re-assert the CUDA-graph prerequisites at the top of every epoch.

        ``on_fit_start`` alone is not enough:
        :class:`~lightning.pytorch.callbacks.GradientAccumulationScheduler`
        *requires* ``trainer.accumulate_grad_batches`` to still be 1 when
        training starts and only raises it from its own
        ``on_train_epoch_start``, so a once-at-fit-start check is guaranteed to
        read 1 and pass, and accumulation would then silently take effect from
        the scheduled epoch. Lightning runs callback hooks before this module
        hook, so by here the scheduler's value for this epoch is already in
        place.

        Raises:
            RuntimeError: Via :meth:`_check_cuda_graph_prerequisites` when a
                prerequisite stopped holding since ``fit`` began.
        """
        if self.training_config.cuda_graph:
            self._check_cuda_graph_prerequisites()

    def _check_cuda_graph_prerequisites(self) -> None:
        """Refuse ``cuda_graph=True`` on a run whose semantics it would change.

        Each of these would otherwise mistrain silently rather than crash, so
        they are hard refusals, not warnings. Checked in
        config-then-environment order so a misconfiguration is reported even on
        a machine with no GPU to run on. Called from ``on_fit_start`` and again
        from :meth:`on_train_epoch_start`, since a callback can move
        ``accumulate_grad_batches`` after ``fit`` has begun.

        Raises:
            RuntimeError: If precision is not ``'32-true'``; if the run is
                distributed (capture and DDP's gradient hooks conflict — DDP
                stashes ``AccumulateGrad`` references); if
                ``accumulate_grad_batches > 1`` (Lightning zeroes gradients only
                on the first micro-batch and expects the rest to accumulate, but
                every replay zeroes them, and the loss is not scaled by the
                accumulation factor); or if the accelerator is not CUDA.
        """
        trainer = self.trainer
        if trainer.precision != "32-true":
            reason = (
                "its GradScaler scales the loss in the precision plugin's pre_backward "
                "hook, which a captured backward pass never runs, so gradients would be "
                "silently unscaled"
                if trainer.precision == "16-mixed"
                else "the captured region's dtypes and autocast state are fixed at capture "
                "time and this path is only validated for full fp32 (bf16 is a measured "
                "regression on this project's architectures besides)"
            )
            raise RuntimeError(
                f"training_config.cuda_graph=True requires trainer.precision='32-true', "
                f"got {trainer.precision!r}: {reason}. Set trainer.precision to 32-true, "
                f"or cuda_graph to false."
            )
        if trainer.world_size > 1:
            raise RuntimeError(
                f"training_config.cuda_graph=True does not support distributed training "
                f"(trainer.world_size={trainer.world_size}). DDP stashes references to "
                f"the autograd graph's AccumulateGrad nodes and inserts gradient "
                f"all-reduce hooks, neither of which survives capture. Train on one "
                f"device, or set cuda_graph to false."
            )
        if trainer.accumulate_grad_batches > 1:
            raise RuntimeError(
                f"training_config.cuda_graph=True does not support gradient accumulation "
                f"(trainer.accumulate_grad_batches={trainer.accumulate_grad_batches}). "
                f"Every replay zeroes the gradients, so nothing would accumulate across "
                f"micro-batches. Raise the loader's batch_size instead, or set cuda_graph "
                f"to false."
            )
        if self.device.type != "cuda":
            raise RuntimeError(
                f"training_config.cuda_graph=True requires a CUDA device, but this module "
                f"is on {self.device!r}. Set trainer.accelerator='cuda', or cuda_graph to "
                f"false."
            )

    def on_train_start(self) -> None:
        """Register val metric tags with TensorBoard's HParams tab.

        Replaces Lightning's default ``hp_metric`` placeholder (disabled via
        ``default_hp_metric=False`` on the logger) with the real validation
        metrics this module emits. Placeholder values are ``0.0``; TensorBoard's
        HParams plugin then fills the columns from the matching scalar tags
        logged by :meth:`validation_step`.
        """
        tb_loggers = [
            logger
            for logger in self.loggers
            if isinstance(logger, lightning.pytorch.loggers.TensorBoardLogger)
        ]
        if not tb_loggers:
            return
        metrics = {
            **{f"psnr/val/{k}": 0.0 for k in self.eval_config.psnr_keys},
            **{f"ssim/val/{k}": 0.0 for k in self.eval_config.ssim_keys},
            **{f"{name}/val": 0.0 for name in self.eval_config.perceptual_keys},
        }
        for tb in tb_loggers:
            tb.log_hyperparams(self._tb_hparams, metrics)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Inject a ``sisr_meta`` provenance key into every saved ``.ckpt``.

        Public, supported Lightning hook (unlike the private ``_save_checkpoint`` the
        bare-weights ``SRWeightsCheckpoint`` callback must override instead). Adding a new
        top-level key is safe for ``--ckpt_path`` resumption: ``LightningCLI._parse_ckpt_path``
        reads only the ``hyper_parameters`` key to seed the model config, so ``sisr_meta``
        is inert there. ``global_step``/``epoch`` are read back from ``checkpoint`` itself
        (populated by Lightning before this hook runs) rather than ``self.trainer``, so the
        metadata always describes the exact step being persisted. No ``monitor``/
        ``monitor_value`` here — a full checkpoint isn't tied to any one monitored metric
        (zero, one, or several ``SRCheckpoint`` callbacks may be watching independently);
        that pairing only exists for :class:`~sisr.training.callbacks.SRWeightsCheckpoint`,
        whose ``monitor`` is a well-defined single value.

        ``batch_step`` comes from the same place, for the same reason: Lightning
        persists ``_batches_that_stepped`` inside the checkpoint's own loop state,
        so reading it here describes the exact step being written rather than
        wherever the live trainer has since got to. It is the axis every logged
        metric uses, and therefore the one that locates this file on a curve —
        ``global_step`` beside it is the optimizer count, which an adversarial
        run advances twice per batch. See
        :meth:`~sisr.training.callbacks._SRCheckpointBase._monitor_candidates`
        for the matching correction to the *filename*.

        Args:
            checkpoint: The checkpoint dict Lightning is about to write to disk; mutated
                in place to add ``checkpoint["sisr_meta"]``.
        """
        epoch_loop = (
            checkpoint.get("loops", {}).get("fit_loop", {}).get("epoch_loop.state_dict", {})
        )
        checkpoint["sisr_meta"] = build_metadata(
            self,
            global_step=checkpoint.get("global_step"),
            batch_step=epoch_loop.get("_batches_that_stepped"),
            epoch=checkpoint.get("epoch"),
        )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op so Lightning iterates ``trainer.test_dataloaders``.

        All metric computation, per-image logging, and image-strip
        emission for the test sets happens in
        :class:`~sisr.training.callbacks.BenchmarkImageLogger.on_test_*`.

        Args:
            batch: ``(lr_img, hr_img)`` tuple from the test loader
                (unused).
            batch_idx: Index of the batch within the current test epoch
                (unused).
            dataloader_idx: Index of the test loader (unused).
        """
        return None

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Run the HR-free inference pipeline: extract → model → reconstruct.

        Shares :meth:`_forward_lr` with :meth:`_forward_sr` (the pipeline
        backing :meth:`_step` / :meth:`predict_rgb`) — the same colorspace
        pipeline minus the HR-dependent center-crop and scoring, neither of
        which has meaning without an HR reference. Paired with
        :meth:`~sisr.training.SRDataModule.predict_dataloader` and typically
        consumed by :class:`~sisr.training.callbacks.SRPredictionWriter`.

        Args:
            batch: LR batch, RGB ``float32`` in ``[0, 1]``, shape
                ``(B, 3, H, W)``, as produced by
                :class:`~sisr.datasets.predict.PredictDataset`.
            batch_idx: Index of the batch within the predict run (unused).
            dataloader_idx: Index of the predict loader (unused — a single
                predict loader is expected).

        Returns:
            SR RGB tensor, ``float32`` in ``[0, 1]``, shape
            ``(B, 3, H', W')`` — ``H'=H*scale``/``W'=W*scale`` for
            SRResNet; for SRCNN, same size as the input only when
            ``padding='same'``, since the default ``'valid'`` (or an
            explicit int) shrinks H/W per conv layer (see
            :meth:`~sisr.models.srcnn.model.SRCNN.forward`).
        """
        _, sr_rgb = self._forward_lr(batch)
        return sr_rgb

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build optimizer (and optional scheduler) from top-level YAML.

        Uniform LR by default — ``self.optimizer(self.parameters())``
        exactly matches what LightningCLI's ``auto_configure_optimizers``
        would do.

        When ``training_config.layer_lrs`` is set, builds per-``Conv2d``
        ``param_groups`` with explicit absolute LRs. This requires every
        trainable parameter to live inside a ``Conv2d`` (SRCNN-style — no
        BatchNorm / PReLU); the validation below makes that explicit.

        Returns:
            The constructed optimizer, or a ``([optimizer], [scheduler])``
            tuple if ``self.lr_scheduler`` is set. Lightning accepts both.

        Raises:
            ValueError: If ``training_config.layer_lrs`` length does not
                match the model's ``Conv2d`` count, or if any trainable
                parameter lives outside a ``Conv2d`` while ``layer_lrs``
                is set.
        """
        lrs = self.training_config.layer_lrs
        if lrs is None:
            optimizer = self.optimizer(self.parameters())
        else:
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            if len(conv_layers) != len(lrs):
                raise ValueError(
                    f"training_config.layer_lrs length {len(lrs)} != "
                    f"Conv2d count {len(conv_layers)}"
                )
            conv_params = {id(p) for layer in conv_layers for p in layer.parameters()}
            other = [
                n
                for n, p in self.model.named_parameters()
                if p.requires_grad and id(p) not in conv_params
            ]
            if other:
                raise ValueError(
                    f"training_config.layer_lrs requires every trainable param to live in "
                    f"a Conv2d; these do not: {other}. Set layer_lrs=None for models with "
                    f"non-Conv layers (BatchNorm, PReLU, etc.)."
                )
            param_groups = [
                {"params": list(layer.parameters()), "lr": lr}
                for layer, lr in zip(conv_layers, lrs, strict=False)
            ]
            optimizer = self.optimizer(param_groups)

        if self.lr_scheduler is None:
            return optimizer
        scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [scheduler]
