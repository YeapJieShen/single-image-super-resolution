"""Adversarial Lightning wrapper for SRGAN — :class:`SRGANLightning`.

Everything but the optimization surface is inherited from
:class:`~sisr.training.lightning_module.SRLightning`: the forward pipeline,
the colorspace handling, validation metrics and checkpoint provenance are the
generator's and are unchanged by training it adversarially. What this subclass
adds is the second network, the second optimizer, and the alternating step that
drives them.

Reference: Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network (https://arxiv.org/pdf/1609.04802), Section 3.2; the
alternation follows Goodfellow et al. (2014) Algorithm 1.
"""

from typing import Any

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ..losses import AdversarialLoss
from ..models.base import SRModel
from ..models.srgan import SRDiscriminator, SRGANEvalConfig, SRGANTrainingConfig
from ..processors import SRProcessor
from .config import SREvalConfig
from .lightning_module import SRLightning


class SRGANLightning(SRLightning):
    """SRResNet generator + SRGAN discriminator, trained by alternating updates.

    Two optimizers cannot be driven by Lightning's automatic loop, so this
    module sets ``automatic_optimization = False`` and owns the whole
    ``{zero_grad, backward, step}`` sequence for both networks (see
    :meth:`training_step`). One consequence is worth stating up front:
    ``trainer.global_step`` counts **optimizer** steps, and this module takes
    two per batch, so after ``N`` batches it reads ``N + N // k`` rather than
    ``N``. Every step-denominated knob (``max_steps``, ``val_check_interval``,
    checkpoint filenames) is in that unit — a ``max_steps`` copied from the
    SRResNet template therefore trains for roughly half the batches.

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
            read by :meth:`training_step` — and refuses ``cuda_graph``.
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
            ``processor.model_channels``.
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
        """
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
        """Refuse distributed training, then run the base's compile warm-up.

        Manual optimization opts out of Lightning's gradient synchronisation:
        with ``automatic_optimization = False`` the strategy's backward no
        longer wraps the step in DDP's ``no_sync``/reducer handling, so the two
        networks' gradients would be reduced at whatever points DDP's hooks
        happen to fire across an alternating schedule where one of the two
        optimizers is idle on most ranks' k-th batches. Supporting it means
        writing that synchronisation explicitly, so this refuses rather than
        training silently out of sync.

        The base's CUDA-graph path is unreachable from here — it is gated on
        ``training_config.cuda_graph``, which :class:`SRGANTrainingConfig`
        refuses at construction — so what ``super()`` contributes is the
        ``torch.compile`` warm-up and the graph-state reset.

        Raises:
            RuntimeError: If ``trainer.world_size > 1``.
        """
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

    def _extra_probe(self, lr: torch.Tensor, hr: torch.Tensor, source: str) -> None:
        """Check the discriminator's declared input size against the real HR crop.

        Args:
            lr: LR sample, ``(C, H, W)``. Unused — the discriminator only ever
                sees HR-sized tensors.
            hr: HR sample, ``(C, H, W)``.
            source: Config path the sample came from, for the error message.

        Raises:
            ValueError: If the HR patch is not square at the declared
                ``hr_input_size``.
        """
        declared = self.discriminator.hparams["hr_input_size"]
        actual = tuple(hr.shape[-2:])
        if actual == (declared, declared):
            return
        raise ValueError(
            f"discriminator hr_input_size={declared} does not match the HR patch "
            f"data.{source} serves ({actual[0]}x{actual[1]}). The discriminator's "
            f"dense head fixes its input size, so a mismatch is a raw Linear shape "
            f"error on the first step otherwise. Set hr_crop_size={declared} on the "
            f"train dataset, or hr_input_size={actual[0]} on the discriminator."
        )
