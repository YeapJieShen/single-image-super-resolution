"""Adversarial Lightning wrapper for SRGAN — :class:`SRGANLightning`.

Everything but the optimization surface is inherited from
:class:`~sisr.training.lightning_module.SRLightning`: the forward pipeline,
the colorspace handling and the validation metrics are the generator's and are
unchanged by training it adversarially. What this subclass adds is the second
network, the second optimizer, the alternating step that drives them, the
generator's optional MSE initialisation, and the adversarial half of the
checkpoint's provenance.

Reference: Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network (https://arxiv.org/pdf/1609.04802), Section 3.2; the
alternation follows Goodfellow et al. (2014) Algorithm 1.
"""

from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_info

from .. import artifacts
from ..losses import AdversarialLoss
from ..models.base import SRModel
from ..models.srgan import SRDiscriminator, SRGANEvalConfig, SRGANTrainingConfig
from ..processors import SRProcessor
from .config import SREvalConfig
from .lightning_module import SRLightning
from .metadata import _class_path, build_metadata

#: ``(section, key)`` of every ``build_metadata`` field ``init_from`` validates —
#: the properties that decide whether one run's generator weights mean anything in
#: another. Reported dotted (``io.scale``) so the message names the YAML the user
#: has to change. Compared against ``build_metadata(self)``, the same builder that
#: wrote the file, so the two sides cannot drift.
_INIT_FROM_FIELDS: tuple[tuple[str, str], ...] = (
    ("model", "class_path"),
    ("model", "init_args"),
    ("processor", "class_path"),
    ("io", "output_range"),
    ("io", "scale"),
)


class SRGANLightning(SRLightning):
    """SRResNet generator + SRGAN discriminator, trained by alternating updates.

    Two optimizers cannot be driven by Lightning's automatic loop, so this
    module sets ``automatic_optimization = False`` and owns the whole
    ``{zero_grad, backward, step}`` sequence for both networks (see
    :meth:`training_step`). One consequence is worth stating up front:
    ``trainer.global_step`` counts **optimizer** steps, and this module takes
    two per batch, so after ``N`` batches it reads ``N + N // k`` rather than
    ``N``. Every global-step knob (``max_steps``, checkpoint filenames,
    ``every_n_train_steps``) is in that unit — a ``max_steps`` copied from the
    SRResNet template therefore trains for roughly half the batches.
    ``val_check_interval`` is *not*: Lightning counts it in batches whenever
    ``check_val_every_n_epoch`` is ``None``, as does this module's own
    scheduler stepping (:meth:`on_train_batch_end`).

    Args:
        model: The generator — an initialised :class:`SRModel` subclass,
            normally :class:`~sisr.models.srresnet.SRResNet`. Required.
        processor: An :class:`SRProcessor` subclass. The discriminator scores
            the generator's output in this processor's *model output* space
            (``[-1, 1]`` under
            :class:`~sisr.processors.RGBSignedOutputProcessor`), not display
            range. Required.
        discriminator: The critic, normally
            :class:`~sisr.models.srgan.SRDiscriminator`. Emits logits; it is
            paired with an ``adversarial_loss`` that applies the sigmoid
            itself. Required.
        training_config: Defaults to :class:`SRGANTrainingConfig`, which
            supplies ``adversarial_weight`` and ``d_steps_per_g_step`` — both
            read by :meth:`training_step` — plus ``init_from`` (read by
            :meth:`setup`, on ``fit`` only).
        eval_config: Defaults to :class:`SRGANEvalConfig` (SRResNet's scoring
            plus perceptual metrics, which are the only metrics that track what
            an adversarial objective optimises).
        criterion: The generator's **content** loss, e.g.
            :class:`torch.nn.MSELoss` or
            :class:`~sisr.losses.VGG19FeatureLoss`. Defaults to
            :class:`torch.nn.MSELoss`. The adversarial term is added on top,
            weighted by ``training_config.adversarial_weight``.
        optimizer: ``OptimizerCallable`` for the **generator**, from top-level
            YAML ``optimizer:``. Defaults to :class:`torch.optim.Adam`.
        lr_scheduler: ``LRSchedulerCallable`` for the generator's optimizer, or
            ``None``. Stepped per batch by :meth:`on_train_batch_end`.
        adversarial_loss: The GAN objective. Defaults to
            :class:`~sisr.losses.AdversarialLoss` (non-saturating, over logits).
        discriminator_optimizer: ``OptimizerCallable`` for the discriminator.
            Defaults to :class:`torch.optim.Adam`. Separate from ``optimizer``
            so the two networks can be tuned independently.
        discriminator_lr_scheduler: ``LRSchedulerCallable`` for the
            discriminator's optimizer, or ``None``.

    Raises:
        ValueError: If ``discriminator``'s ``in_channels`` disagrees with
            ``processor.model_channels``. ``training_config.init_from`` is
            validated in :meth:`setup` instead — construction touches no files.
        TypeError: Via :class:`SRLightning` if ``model`` / ``processor`` are
            not of the required base types.
    """

    def __init__(
        self,
        model: SRModel,
        processor: SRProcessor,
        discriminator: SRDiscriminator,
        training_config: SRGANTrainingConfig | None = None,
        eval_config: SREvalConfig | None = None,
        criterion: torch.nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        adversarial_loss: AdversarialLoss | None = None,
        discriminator_optimizer: OptimizerCallable = torch.optim.Adam,
        discriminator_lr_scheduler: LRSchedulerCallable | None = None,
    ):
        super().__init__(
            model=model,
            processor=processor,
            training_config=training_config or SRGANTrainingConfig(),
            eval_config=eval_config or SRGANEvalConfig(),
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        # The type hint is not enforced at runtime, and everything this module
        # asserts about its config rests on the subclass: the base carries no
        # adversarial_weight/d_steps_per_g_step.
        if not isinstance(self.training_config, SRGANTrainingConfig):
            raise TypeError(
                f"training_config must be an SRGANTrainingConfig, got "
                f"{type(self.training_config).__name__}. This module reads "
                f"adversarial_weight and d_steps_per_g_step off it on every step."
            )

        # The correlated check SRGANTrainingConfig.validate_against cannot do:
        # its (model, processor) signature never sees the discriminator.
        if discriminator.hparams["in_channels"] != processor.model_channels:
            raise ValueError(
                f"SRDiscriminator in_channels={discriminator.hparams['in_channels']} does "
                f"not match {type(processor).__name__}.model_channels="
                f"{processor.model_channels}. The discriminator scores the generator's "
                f"output in model space, so it must accept the same channel count the "
                f"generator emits — otherwise the first step fails as a raw Conv2d shape "
                f"mismatch."
            )

        self.discriminator = discriminator
        # `is not None`, not `or`: a container-backed loss (nn.ModuleList with no
        # entries yet) is falsy and would be silently swapped for the default.
        self.adversarial_loss = (
            adversarial_loss if adversarial_loss is not None else AdversarialLoss()
        )
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_lr_scheduler = discriminator_lr_scheduler
        self.automatic_optimization = False

    def setup(self, stage: str | None = None) -> None:
        """Run the base's sample probe, then MSE-initialise the generator — ``fit`` only.

        The load is gated on the stage because *every* subcommand instantiates
        the model from config, ``--ckpt_path`` included: an ``__init__``-time
        read made ``validate``/``test``/``export`` depend on an artifact they
        never use, so they died with ``FileNotFoundError`` on any machine
        without it (every clone, every worktree, CI).

        Here rather than in :meth:`on_fit_start` because Lightning restores a
        ``--ckpt_path`` resume *between* the two hooks (``Trainer._run``): from
        the later one this load would overwrite the resumed generator with the
        pretrained weights and silently restart it, while the discriminator and
        the optimizer state carried on from where they were.

        On a ``fit --ckpt_path`` resume, this runs (``Trainer._run`` calls the
        setup hook at line 1039) *before* the resume restores the module's own
        state (line 1046), so the resumed generator wins and the final model
        state is correct either way. The costs are one wasted weights read and
        a hard dependency on ``init_from``'s artifact existing on the resuming
        box, even though its result is about to be discarded. The upside: the
        five metadata refusals below still run on every resume, so a config
        that has drifted from the run being resumed (a changed architecture,
        processor, or scale) is still caught rather than only checked on a
        fresh ``fit``.

        Args:
            stage: Lightning trainer stage — ``'fit'``, ``'validate'``,
                ``'test'`` or ``'predict'``. Only ``'fit'`` loads.

        Raises:
            ValueError: Via :meth:`_init_generator_from`, if
                ``training_config.init_from`` names an artifact this run's
                generator cannot be initialised from.
        """
        super().setup(stage)
        # Last: after the base's own probe, and after every construction-time
        # refusal — a load is the one step here that touches the filesystem.
        if stage == "fit" and self.training_config.init_from is not None:
            self._init_generator_from(self.training_config.init_from)

    def _init_generator_from(self, init_from: str) -> None:
        """Initialise the **generator only** from a bare-weights artifact.

        Ledig et al. scope the MSE-initialisation trick to "when training the
        actual GAN", so a paper-faithful SRGAN run starts from an MSE-trained
        SRResNet rather than from scratch. The discriminator is not in that file
        and is not touched: the adversarial half starts fresh by design.

        Every mismatch is refused rather than warned about. Weights trained
        under a different architecture, processor, output range or scale load
        into a model that then trains and scores without ever erroring — only
        the numbers are wrong, which is indistinguishable from a bad run.

        Args:
            init_from: Path to a ``.safetensors`` written by
                :class:`~sisr.training.callbacks.SRWeightsCheckpoint` with
                ``attribute='model'``.

        Raises:
            ValueError: If ``init_from`` is the string ``'None'`` a CLI
                ``=null`` collapses to; if it names a ``.ckpt`` or another
                network's component file; if the payload carries no ``meta``
                to validate against; or if any of :data:`_INIT_FROM_FIELDS`
                disagrees with this run.
            RuntimeError: From ``load_state_dict(strict=True)`` if the metadata
                matches but the tensors do not.
        """
        if init_from == "None":
            raise ValueError(
                "init_from is the literal string 'None', not a path. jsonargparse "
                "coerces a null CLI override of a `str | None` field to that string, so "
                "--...init_from=null does not unset init_from — it renames the file "
                "being looked for. Pass an overlay instead: --config a YAML setting "
                "init_from: null."
            )

        path = Path(init_from)
        if path.suffix == ".ckpt":
            raise ValueError(
                f"init_from expects a bare-weights artifact, but got the resumable checkpoint "
                f"{path.name!r}. A .ckpt holds the whole LightningModule under "
                f"'model.'-prefixed keys (plus optimizer state, and no per-network "
                f"provenance to validate), so loading it into the bare generator is an "
                f"unreadable key dump at best. Point init_from at the sibling "
                f"sr-weights file that SRWeightsCheckpoint writes alongside it, or resume "
                f"the whole run with --ckpt_path instead."
            )

        tensors, meta = artifacts.load(path)

        # Only when 'kind' is present: it is an additive field, and the golden
        # artifact the template ships predates it. Absent means generator.
        if meta.get("kind") == "component":
            name = meta.get("component", {}).get("name", "unnamed")
            raise ValueError(
                f"init_from={path.name!r} holds this run's {name!r} component, not a "
                f"generator: an SRWeightsCheckpoint with attribute={name!r} wrote it, and "
                f"the SRGAN template writes both kinds into one dirpath. Its metadata "
                f"describes only that component, so none of the generator checks "
                f"(architecture, processor, output range, upscaling factor) can run "
                f"against it. Point init_from at the sr-weights sibling."
            )

        # Stricter than the module default on purpose: initialising a generator from
        # the wrong weights is not merely unlabelled, it silently mistrains.
        artifacts.require_compatible(
            meta,
            build_metadata(self),
            fields=_INIT_FROM_FIELDS,
            source=f"init_from={path.name!r}",
        )

        self.model.load_state_dict(tensors, strict=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build the generator and discriminator optimizers, in that order.

        The generator's optimizer is built from ``self.model.parameters()``,
        **never** ``self.parameters()``: the latter includes the discriminator,
        whose parameters would then be stepped to *minimise* the adversarial
        loss — training the discriminator to lose. Nothing fails when that
        happens; both loss curves keep moving and look healthy.

        Returns:
            ``([opt_g, opt_d], schedulers)``, where ``schedulers`` holds the
            generator's then the discriminator's, and is empty when neither is
            set. The shape is always this pair — never a bare optimizer list —
            so callers never branch on it; :meth:`training_step` unpacks the
            optimizers positionally and the order is part of the contract.

        Raises:
            ValueError: If ``training_config.layer_lrs`` is set.
        """
        # Inherited through SRResNetTrainingConfig, so YAML can set it; this
        # override never reads it, and a silent uniform-LR fallback is the one
        # unsignalled misconfiguration in a module that refuses four others.
        if self.training_config.layer_lrs is not None:
            raise ValueError(
                f"training_config.layer_lrs is not supported for adversarial training "
                f"(got {len(self.training_config.layer_lrs)} entries). SRLightning's "
                f"per-Conv2d param_groups are built by the base configure_optimizers, "
                f"which this override replaces to build one optimizer per network, so "
                f"the setting would be silently ignored and both networks would train "
                f"at their optimizer's uniform lr. Set layer_lrs to null and tune "
                f"optimizer.lr / discriminator_optimizer.lr instead."
            )
        opt_g = self.optimizer(self.model.parameters())
        opt_d = self.discriminator_optimizer(self.discriminator.parameters())
        schedulers = []
        if self.lr_scheduler is not None:
            schedulers.append(self.lr_scheduler(opt_g))
        if self.discriminator_lr_scheduler is not None:
            schedulers.append(self.discriminator_lr_scheduler(opt_d))
        return [opt_g, opt_d], schedulers

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """One Goodfellow outer iteration: a discriminator step, plus every k-th a generator one.

        Order is discriminator-first (Algorithm 1), so the generator is updated
        against the discriminator that just moved. ``k``
        (``training_config.d_steps_per_g_step``) is spent *across* batches
        rather than looped here, giving each discriminator step a fresh
        minibatch as Algorithm 1 specifies.

        One generator forward per batch, reused: the discriminator's backward
        runs on ``sr.detach()`` so it never touches the generator's graph, and
        the generator's backward then flows through a second, fresh
        discriminator forward. No ``retain_graph`` is needed anywhere.

        Args:
            batch: ``(lr_img, hr_img)`` tuple from the train loader. Both RGB,
                ``float32`` in ``[0, 1]``.
            batch_idx: Index of the batch within the current epoch. Also the
                alternation counter, so ``k`` restarts its cycle each epoch.

        Returns:
            ``None``: under manual optimization Lightning does not consume a
            returned loss.
        """
        lr_img, hr_img = batch
        opt_g, opt_d = self.optimizers()

        sr, _, hr_cropped = self._forward_sr(lr_img, hr_img, need_sr_rgb=False)
        hr_for_loss = self.processor.extract_target(hr_cropped)

        self.toggle_optimizer(opt_d)
        d_loss = self.adversarial_loss.discriminator_loss(
            self.discriminator(hr_for_loss), self.discriminator(sr.detach())
        )
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)
        self.log("loss/train/d", d_loss, on_step=True, prog_bar=True)

        if (batch_idx + 1) % self.training_config.d_steps_per_g_step:
            return

        # toggle_optimizer flips requires_grad off for every parameter outside
        # opt_g, so the generator's backward — which must pass THROUGH the
        # discriminator to reach the generator — stops accumulating into the
        # discriminator's .grad on the way. Measured: without it, that backward
        # changes 34 of the discriminator's gradient tensors. It survives today
        # only because opt_d.zero_grad() above clears the debris before the next
        # discriminator backward, which is an emergent property of the current
        # ordering rather than anything stated. The generator still receives its
        # gradient either way (also measured).
        self.toggle_optimizer(opt_g)
        content = self.criterion(sr, hr_for_loss)
        adversarial = self.adversarial_loss.generator_loss(self.discriminator(sr))
        g_loss = content + self.training_config.adversarial_weight * adversarial
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.log("loss/train", g_loss, prog_bar=True, on_step=True)
        self.log("loss/train/content", content, on_step=True)
        self.log("loss/train/adv", adversarial, on_step=True)
        self._log_loss_terms("train")

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Step the LR schedulers by hand — manual optimization does not.

        Milestones are therefore counted in **batches**, which is the correct
        unit for the paper's schedule (1e5 iterations at 1e-4, then 1e5 at
        1e-5) and deliberately *not* the unit ``trainer.global_step`` reports
        (see the class docstring).

        Args:
            outputs: Whatever :meth:`training_step` returned — always ``None``
                here. Unused.
            batch: The batch just processed. Unused.
            batch_idx: Index of the batch within the current epoch. Unused —
                every batch steps the schedulers, including the ones that skip
                the generator, since the discriminator stepped regardless.
        """
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        # lr_schedulers() returns the scheduler ITSELF when exactly one is
        # configured, and a list only when there are several.
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    def on_fit_start(self) -> None:
        """State the step-axis conversion, refuse distributed training, then warm up.

        Manual optimization opts out of Lightning's gradient synchronisation:
        with ``automatic_optimization = False`` the strategy's backward no
        longer wraps the step in DDP's ``no_sync``/reducer handling, so the two
        networks' gradients would be reduced at whatever points DDP's hooks
        happen to fire across an alternating schedule where one of the two
        optimizers is idle on most ranks' k-th batches. Supporting it means
        writing that synchronisation explicitly, so this refuses rather than
        training silently out of sync.

        What ``super()`` contributes is the ``torch.compile`` warm-up.

        Raises:
            RuntimeError: If ``trainer.world_size > 1``.
        """
        # Lightning's own knobs cannot be renamed, and under this paradigm they
        # count different things: max_steps and every_n_train_steps are optimizer
        # steps, everything else is batches. Documentation has not been enough --
        # a max_steps copied from the SRResNet template trains half as long -- so
        # the conversion is stated once, against this run's actual numbers.
        max_steps = self.trainer.max_steps
        if max_steps and max_steps > 0:
            rank_zero_info(
                f"SRGAN takes 2 optimizer steps per batch, so max_steps={max_steps} is "
                f"{max_steps // 2} batches. val_check_interval, every_n_batches and the "
                f"hand-stepped scheduler milestones all count batches; max_steps and "
                f"every_n_train_steps count optimizer steps."
            )

        if self.trainer.world_size > 1:
            raise RuntimeError(
                f"SRGANLightning does not support distributed training "
                f"(trainer.world_size={self.trainer.world_size}). It trains under manual "
                f"optimization, which opts out of Lightning's automatic gradient "
                f"synchronisation, so the generator's and discriminator's gradients would "
                f"not be reduced across ranks — each rank would train its own divergent "
                f"pair with nothing failing. Train on one device."
            )
        super().on_fit_start()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Append the adversarial half of this run's provenance.

        The base builds ``sisr_meta`` around ``module.model`` — still correct,
        since the generator is the distributable artifact. These blocks are
        added only to the ``.ckpt``: the bare ``.pt`` written by
        :class:`~sisr.training.callbacks.SRWeightsCheckpoint` holds one
        network's weights and gets metadata describing *that* network, not this
        run's whole adversarial setup.

        Args:
            checkpoint: The checkpoint dict Lightning is about to write;
                mutated in place.
        """
        super().on_save_checkpoint(checkpoint)
        checkpoint["sisr_meta"]["discriminator"] = {
            "class_path": _class_path(self.discriminator),
            "init_args": dict(self.discriminator.hparams),
        }
        checkpoint["sisr_meta"]["adversarial"] = {
            "loss": _class_path(self.adversarial_loss),
            "weight": self.training_config.adversarial_weight,
            "d_steps_per_g_step": self.training_config.d_steps_per_g_step,
        }

    def _extra_probe(self, lr: torch.Tensor, hr: torch.Tensor, source: str) -> None:
        """Check the discriminator's declared input size against the HR crop it will score.

        Keys on the **post-crop** size, not the raw HR patch: what
        :meth:`training_step` feeds the discriminator is
        ``extract_target(hr_cropped)``, and ``_forward_sr`` center-crops HR to
        the generator's output size. The two coincide only while the generator
        emits exactly ``scale x lr``, which is an SRResNet property rather than
        anything this module requires.

        Scoped to the **train** dataset. ``hr_input_size`` constrains the
        training crop and nothing else — a ``validate``/``test`` run hands
        :meth:`~sisr.training.lightning_module.SRLightning.setup` a full image,
        which would both fail this check for no reason and pay a
        full-resolution generator forward on CPU at setup.

        The probe forward puts the **whole module** in ``eval`` mode under
        ``no_grad``, not just ``self.model``: the forward selector reads
        ``self.training`` (the LightningModule's flag), so ``self.model.eval()``
        alone would still route a compiled run through ``self._compiled`` and
        trigger a dynamo compile here — before, and at a different shape from,
        the deliberate :meth:`on_fit_start` warm-up. ``self.eval()`` recurses
        into ``self.model``, so the reason the eval-mode forward existed in the
        first place still holds: a train-mode forward would fold this sample
        into the generator's BatchNorm running statistics before training has
        taken a single step.

        Args:
            lr: LR sample, ``(C, H, W)``.
            hr: HR sample, ``(C, H, W)``.
            source: Config path the sample came from, for the error message.

        Raises:
            ValueError: If the cropped HR the discriminator would score is not
                square at the declared ``hr_input_size``.
        """
        if source != "train_dataset":
            return
        declared = self.discriminator.hparams["hr_input_size"]
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _, _, hr_cropped = self._forward_sr(lr[None], hr[None], need_sr_rgb=False)
        finally:
            self.train(was_training)
        actual = tuple(self.processor.extract_target(hr_cropped).shape[-2:])
        if actual == (declared, declared):
            return
        raise ValueError(
            f"discriminator hr_input_size={declared} does not match the HR crop it "
            f"would score ({actual[0]}x{actual[1]}) for the samples data.{source} "
            f"serves ({tuple(hr.shape[-2:])} before {type(self.model).__name__}'s "
            f"output size crops it). The discriminator's dense head fixes its input "
            f"size, so a mismatch is a raw Linear shape error on the first step "
            f"otherwise. Set hr_input_size={actual[0]} on the discriminator, or size "
            f"the dataset's crops so the generator emits {declared}x{declared}."
        )
