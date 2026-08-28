"""SRGANLightning — the two-optimizer alternating training loop.

Every failure mode this file guards is silent: nothing crashes, both loss
curves keep moving, and the model trains wrongly. So each test drives a real
``Trainer`` loop and asserts on parameters, gradients or step counts, not on
whether a call happened.
"""

from types import SimpleNamespace

import lightning
import pytest
import safetensors.torch
import torch

from sisr import artifacts
from sisr.losses import AdversarialLoss, VGG19FeatureLoss
from sisr.models.base import SRModel
from sisr.models.srgan import SRDiscriminator, SRGANEvalConfig, SRGANTrainingConfig
from sisr.models.srresnet import SRResNet, SRResNetTrainingConfig
from sisr.processors import RGBSignedOutputProcessor, YChannelProcessor
from sisr.training import SRCheckpoint, SRGANLightning, SRLightning, SRWeightsCheckpoint
from sisr.training.config import SRTrainingConfig
from sisr.training.metadata import build_component_metadata, build_metadata

# Every fit below runs on CPU with a single-worker loader over a handful of
# batches, which Lightning flags as three PossibleUserWarnings the strict global
# filterwarnings=error would otherwise fail on. Suppressed by message rather than
# by category: PossibleUserWarning is also how Lightning reports real
# misconfigurations, and blanket-ignoring the class hides those too.
_ignore_cpu_fit_warnings = pytest.mark.filterwarnings(
    "ignore:GPU available but not used:lightning.pytorch.utilities.warnings.PossibleUserWarning",
    "ignore:The '.*' does not have many workers:"
    "lightning.pytorch.utilities.warnings.PossibleUserWarning",
    "ignore:You defined a `validation_step` but have no `val_dataloader`:"
    "lightning.pytorch.utilities.warnings.PossibleUserWarning",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def build_gan_module(
    k=1,
    hr_input_size=96,
    num_residual_blocks=1,
    lr_scheduler=None,
    discriminator_lr_scheduler=None,
    cls=SRGANLightning,
    criterion=None,
    adversarial_weight=SRGANTrainingConfig.adversarial_weight,
    layer_lrs=None,
    init_from=None,
):
    """A minimal but real SRGANLightning — one residual block keeps it CPU-fast."""
    return cls(
        model=SRResNet(scale=4, num_residual_blocks=num_residual_blocks),
        processor=RGBSignedOutputProcessor(),
        discriminator=SRDiscriminator(hr_input_size=hr_input_size),
        adversarial_loss=AdversarialLoss(),
        criterion=torch.nn.MSELoss() if criterion is None else criterion,
        training_config=SRGANTrainingConfig(
            d_steps_per_g_step=k,
            example_input_shape=(3, 24, 24),
            adversarial_weight=adversarial_weight,
            layer_lrs=layer_lrs,
            init_from=init_from,
        ),
        eval_config=SRGANEvalConfig(perceptual_metrics=[]),
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        discriminator_optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        lr_scheduler=lr_scheduler,
        discriminator_lr_scheduler=discriminator_lr_scheduler,
    )


def build_source_module():
    """The realistic ``init_from`` source: a plain, MSE-trained SRResNet run.

    Its generator and processor match :func:`build_gan_module`'s exactly, so a
    ``.pt`` written from it is the one artifact ``init_from`` must accept.
    """
    return SRLightning(
        model=SRResNet(scale=4, num_residual_blocks=1),
        processor=RGBSignedOutputProcessor(),
        training_config=SRResNetTrainingConfig(),
    )


def write_weights(tmp_path, **meta_overrides):
    """A generator-weights ``.pt`` whose meta can be corrupted one field at a time.

    Written with the same :func:`~sisr.training.metadata.build_metadata` that
    ``SRWeightsCheckpoint`` uses, so what is exercised is the real payload shape.
    """
    source = build_source_module()
    meta = build_metadata(source)
    for dotted, value in meta_overrides.items():
        section, key = dotted.split(".")
        meta[section][key] = value
    path = tmp_path / "sr-weights.safetensors"
    artifacts.save(path, source.model.state_dict(), meta)
    return path, source


def init_gan_from(path, **kwargs):
    """Build a GAN module and run the ``fit`` setup — the hook ``init_from`` loads in.

    The load is deferred out of ``__init__`` so that ``validate``/``test``/
    ``export``, which all instantiate the model from config, never read the file.
    Every ``init_from`` assertion therefore has to drive the stage that does.
    """
    module = build_gan_module(init_from=str(path), **kwargs)
    module.setup("fit")
    return module


def clone_params(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def params_changed(before, after):
    return any(not torch.equal(before[name], value) for name, value in after.items())


def grad_snapshot(module):
    return {
        name: (None if p.grad is None else p.grad.detach().clone())
        for name, p in module.named_parameters()
    }


def same_grad(before, after):
    if (before is None) != (after is None):
        return False
    return before is None or torch.equal(before, after)


def gan_loader(n_batches, hr=96, scale=4):
    """(lr, hr) pairs matching the discriminator's declared input size."""
    lr_size = hr // scale
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.rand(n_batches, 3, lr_size, lr_size), torch.rand(n_batches, 3, hr, hr)
        ),
        batch_size=1,
    )


def fit_gan(
    module, n_batches, callbacks=None, val_batches=0, enable_checkpointing=False, **trainer_kwargs
):
    """Run a real Trainer loop over ``module`` and hand back the trainer."""
    trainer = lightning.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )
    trainer.fit(
        module,
        gan_loader(n_batches),
        gan_loader(val_batches) if val_batches else None,
    )
    return trainer


class ParamRecorder(lightning.Callback):
    """Snapshots both networks' parameters before training and after every batch.

    Index 0 is the pre-training state, index i+1 the state after batch i.
    """

    def __init__(self):
        self.g = []
        self.d = []

    def on_train_start(self, trainer, pl_module):
        self._record(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._record(pl_module)

    def _record(self, pl_module):
        self.g.append(clone_params(pl_module.model))
        self.d.append(clone_params(pl_module.discriminator))


class BackwardGradSpy(SRGANLightning):
    """Records the discriminator's gradients immediately before every backward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_grads_before_backward = []

    def manual_backward(self, loss, *args, **kwargs):
        self.d_grads_before_backward.append(grad_snapshot(self.discriminator))
        return super().manual_backward(loss, *args, **kwargs)


class ProbeStateSpy(SRGANLightning):
    """Records the module- and generator-level ``training`` flags at probe forward time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_flags = []

    def _forward_sr(self, *args, **kwargs):
        self.probe_flags.append((self.training, self.model.training))
        return super()._forward_sr(*args, **kwargs)


class ShrinkingGenerator(SRModel):
    """A pre-upsampled generator that emits less than the HR patch it is handed.

    ``shrink`` pixels of it, via one valid-padded conv. Exists to separate the
    raw HR patch from the crop the discriminator scores, which SRResNet's exact
    ``scale x lr`` output makes indistinguishable.
    """

    input_contract = "pre_upsampled"

    @property
    def variant_tag(self) -> str:
        return "shrink"

    def __init__(self, channels=3, shrink=16):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=shrink + 1, padding="valid")
        self._hparams = {"in_out_channels": channels, "scale": 4, "shrink": shrink}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ZeroContentLoss(torch.nn.Module):
    """A content term that is identically zero, with an identically-zero gradient.

    Still a function of ``sr``, so the generator stays reachable from ``g_loss``
    and a backward through it is well-formed — it just contributes nothing. That
    leaves the adversarial term as the only thing that can put a non-zero number
    into a generator gradient.
    """

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return sr.sum() * 0.0


def generator_grads_after_one_seeded_batch(adversarial_weight):
    """Generator ``.grad`` after one batch with the content term zeroed.

    Seeded either side of construction so the two runs this is called for differ
    in nothing but ``adversarial_weight``: the discriminator's own step is
    independent of it, so the generator sees an identical discriminator.
    """
    torch.manual_seed(0)
    module = build_gan_module(
        k=1, criterion=ZeroContentLoss(), adversarial_weight=adversarial_weight
    )
    torch.manual_seed(1)
    fit_gan(module, n_batches=1)
    # Nothing clears grads after opt_g.step(), so these are g_loss's.
    return {name: p.grad.detach().clone() for name, p in module.model.named_parameters()}


def _pair_dataset(hr_size, scale):
    lr_size = hr_size // scale
    return torch.utils.data.TensorDataset(
        torch.rand(1, 3, lr_size, lr_size), torch.rand(1, 3, hr_size, hr_size)
    )


def fake_datamodule(hr_crop_size=96, scale=4):
    """Minimal stand-in exposing the read accessors SRLightning.setup probes."""
    return SimpleNamespace(
        train_dataset=_pair_dataset(hr_crop_size, scale),
        val_dataset=None,
        test_datasets=None,
    )


def val_only_datamodule(hr_size=128, scale=4):
    """A ``validate``-style datamodule: no train dataset, and full-image samples.

    ``hr_size`` is deliberately not the discriminator's ``hr_input_size`` —
    validation images are whole pictures, never the training crop.
    """
    return SimpleNamespace(
        train_dataset=None,
        val_dataset=_pair_dataset(hr_size, scale),
        test_datasets=None,
    )


def multistep(milestone=2):
    return lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[milestone], gamma=0.1)


# ---------------------------------------------------------------------------
# optimizers
# ---------------------------------------------------------------------------


def test_generator_optimizer_excludes_discriminator_parameters():
    """The trap: self.parameters() sweeps D into the generator's optimizer,
    which then MINIMISES the adversarial loss w.r.t. D — training D to lose,
    with entirely healthy-looking curves and nothing failing."""
    module = build_gan_module()
    opt_g, opt_d = module.configure_optimizers()[0]

    g_ids = {id(p) for group in opt_g.param_groups for p in group["params"]}
    d_ids = {id(p) for p in module.discriminator.parameters()}
    assert g_ids.isdisjoint(d_ids)
    # ...and each optimizer owns exactly its own network, so the disjointness
    # above cannot be satisfied by an under-populated generator optimizer.
    assert g_ids == {id(p) for p in module.model.parameters()}
    assert {id(p) for group in opt_d.param_groups for p in group["params"]} == d_ids


def test_configure_optimizers_returns_the_optimizer_list_first_in_a_fixed_order():
    """training_step unpacks self.optimizers() positionally, so the order is
    part of the contract; the shape stays a (optimizers, schedulers) pair even
    with no schedulers, so callers never have to branch on it."""
    module = build_gan_module()

    optimizers, schedulers = module.configure_optimizers()

    assert schedulers == []
    opt_g, opt_d = optimizers
    assert {id(p) for group in opt_g.param_groups for p in group["params"]} == {
        id(p) for p in module.model.parameters()
    }
    assert {id(p) for group in opt_d.param_groups for p in group["params"]} == {
        id(p) for p in module.discriminator.parameters()
    }


# ---------------------------------------------------------------------------
# the alternating step
# ---------------------------------------------------------------------------


@_ignore_cpu_fit_warnings
def test_one_step_moves_both_networks_at_k1():
    module = build_gan_module(k=1)
    recorder = ParamRecorder()

    fit_gan(module, n_batches=1, callbacks=[recorder])

    assert params_changed(recorder.g[0], recorder.g[1]), "generator did not update"
    assert params_changed(recorder.d[0], recorder.d[1]), "discriminator did not update"


@_ignore_cpu_fit_warnings
def test_at_k2_the_generator_updates_only_on_every_second_batch():
    """k is spent ACROSS batches (Goodfellow Algorithm 1: each discriminator
    step sees a fresh minibatch), not looped inside one training_step."""
    module = build_gan_module(k=2)
    recorder = ParamRecorder()

    fit_gan(module, n_batches=2, callbacks=[recorder])

    assert not params_changed(recorder.g[0], recorder.g[1]), "G must not update on batch 0 at k=2"
    assert params_changed(recorder.d[0], recorder.d[1]), "D must update every batch"
    assert params_changed(recorder.g[1], recorder.g[2]), "G must update on batch 1 at k=2"
    assert params_changed(recorder.d[1], recorder.d[2]), "D must update every batch"


@_ignore_cpu_fit_warnings
def test_generator_backward_leaves_the_discriminator_gradients_untouched():
    """The generator's backward passes THROUGH the discriminator to reach the
    generator, and accumulates into D's .grad on the way unless toggled off.

    Measured: without toggle_optimizer the generator's backward changes 34 of
    the discriminator's gradient tensors (e.g. features.0.weight by 0.200085);
    with it, none. It is currently harmless only because the next batch's
    opt_d.zero_grad() clears the debris first — an emergent property of the
    ordering, not a stated invariant, and a live bug the moment anything
    reorders the steps or accumulates gradients.

    The property is "the generator's backward does not *change* D's gradients",
    not "D has no gradient afterwards": D legitimately carries the gradient of
    its own loss at that point, since opt_d.zero_grad() runs *before* the
    discriminator's backward and nothing clears it afterwards (measured: 74.99
    on features.0.weight after a correct step). Both halves of the assertion
    matter — one that only checked D would pass if the generator had silently
    stopped learning altogether.
    """
    module = build_gan_module(k=1, cls=BackwardGradSpy)

    fit_gan(module, n_batches=1)

    assert len(module.d_grads_before_backward) == 2, "expected the D backward, then the G backward"
    before_generator_backward = module.d_grads_before_backward[1]
    after = grad_snapshot(module.discriminator)
    polluted = [
        name for name, grad in after.items() if not same_grad(before_generator_backward[name], grad)
    ]
    assert not polluted, f"generator backward polluted discriminator gradients at {polluted[:3]}"
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) for p in module.model.parameters()
    ), "generator must still receive its gradient through the discriminator"


@_ignore_cpu_fit_warnings
def test_the_generator_is_trained_by_the_adversarial_term():
    """What makes this module adversarial rather than a plain SRResNet run.

    Every other test here passes on the content loss alone, so two edits that
    silently drop the adversarial objective — dropping the term from ``g_loss``,
    or scoring ``sr.detach()`` in it, the plausible symmetry with the
    discriminator's line — would go unnoticed while ``loss/train/adv`` kept
    being logged.

    Zeroing the content term is what discriminates: the generator's gradient can
    then only have arrived through the discriminator. The weight is raised off
    the paper's 1e-3 purely so the resulting SGD update is unambiguous in
    float32; the gradient assertion is the load-bearing one.
    """
    module = build_gan_module(k=1, criterion=ZeroContentLoss(), adversarial_weight=1.0)
    recorder = ParamRecorder()

    fit_gan(module, n_batches=1, callbacks=[recorder])

    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) for p in module.model.parameters()
    ), "with the content term zeroed the generator got no gradient — the adversarial term is inert"
    assert params_changed(recorder.g[0], recorder.g[1]), "generator did not update"


@_ignore_cpu_fit_warnings
def test_adversarial_weight_scales_the_generators_gradient():
    """The paper's 1e-3 has to actually multiply the adversarial term.

    Doubling it must double the generator's gradient exactly, since with the
    content term zeroed that gradient IS ``weight * d(adversarial)/d(theta)``.
    A dropped or detached adversarial term leaves both runs at zero and fails
    the non-degeneracy check first.
    """
    half = generator_grads_after_one_seeded_batch(0.5)
    full = generator_grads_after_one_seeded_batch(1.0)

    assert any(torch.count_nonzero(g) for g in full.values()), "no adversarial gradient to scale"
    for name, grad in full.items():
        assert torch.allclose(half[name], grad * 0.5, rtol=1e-5, atol=1e-8), name


@_ignore_cpu_fit_warnings
@pytest.mark.parametrize(("k", "n_batches", "expected"), [(1, 20, 40), (2, 20, 30)])
def test_global_step_counts_optimizer_steps_not_batches(k, n_batches, expected):
    """global_step counts optimizer steps and this module takes two per batch,
    so it is N + N//k — not the batch count. That is the unit max_steps,
    checkpoint filenames and every_n_train_steps use, so a max_steps copied from
    the SRResNet template trains for half the iterations. (val_check_interval is
    the exception: Lightning counts it in batches. Manual optimization is not the
    cause of any of it: with a single optimizer global_step still equals the
    batch count.)
    """
    trainer = fit_gan(build_gan_module(k=k), n_batches=n_batches)

    assert trainer.global_step == expected


@_ignore_cpu_fit_warnings
def test_training_keeps_updating_across_a_validation_boundary():
    """Everything but the optimization surface is inherited, so validation must
    still score and training must still move afterwards. A mid-training
    validation run has frozen this project's training before, and it did so
    with the loss curve still moving."""
    module = build_gan_module(k=1)
    recorder = ParamRecorder()

    trainer = fit_gan(
        module, n_batches=4, val_batches=1, callbacks=[recorder], val_check_interval=2
    )

    assert "loss/val" in trainer.callback_metrics
    assert "psnr/val/RGB" in trainer.callback_metrics
    # Snapshots are [initial, b0, b1, b2, b3]; validation runs between b1 and b2.
    assert params_changed(recorder.g[2], recorder.g[3]), "generator froze after validation"
    assert params_changed(recorder.d[2], recorder.d[3]), "discriminator froze after validation"


# ---------------------------------------------------------------------------
# LR schedulers — manual optimization does not step them
# ---------------------------------------------------------------------------


@_ignore_cpu_fit_warnings
def test_a_single_lr_scheduler_steps_once_per_batch():
    """With exactly one scheduler configured, LightningModule.lr_schedulers()
    returns the scheduler ITSELF, not a one-element list — iterating the bare
    return is a TypeError, so this configuration must be handled explicitly."""
    module = build_gan_module(lr_scheduler=multistep(milestone=2))

    trainer = fit_gan(module, n_batches=3)

    assert len(trainer.lr_scheduler_configs) == 1
    assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == 3
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.01)


@_ignore_cpu_fit_warnings
def test_both_schedulers_step_and_count_milestones_in_batches():
    """Milestones are counted in batches — deliberately a different unit from
    trainer.global_step, which counts this module's two optimizer steps per
    batch. The paper's schedule (1e5 iterations, then 1e5 more) is in batches."""
    module = build_gan_module(
        lr_scheduler=multistep(milestone=2), discriminator_lr_scheduler=multistep(milestone=2)
    )

    trainer = fit_gan(module, n_batches=3)

    assert [c.scheduler.last_epoch for c in trainer.lr_scheduler_configs] == [3, 3]
    assert [o.param_groups[0]["lr"] for o in trainer.optimizers] == pytest.approx([0.01, 0.01])
    assert trainer.global_step == 6


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_discriminator_channels_must_match_the_processor():
    """A 1-channel processor with a 3-channel discriminator is otherwise a raw
    Conv2d shape error on the first step."""
    with pytest.raises(ValueError, match="in_channels"):
        SRGANLightning(
            model=SRResNet(scale=4, in_out_channels=1, num_residual_blocks=1),
            processor=YChannelProcessor(),
            discriminator=SRDiscriminator(in_channels=3),
            training_config=SRGANTrainingConfig(),
        )


def test_unset_components_default_to_the_srgan_paper_ones():
    """A base SREvalConfig here would silently score SRGAN with wang SSIM,
    crop_border=0 and no perceptual metric — the only metric family that tracks
    what an adversarial objective optimises."""
    module = SRGANLightning(
        model=SRResNet(scale=4, num_residual_blocks=1),
        processor=RGBSignedOutputProcessor(),
        discriminator=SRDiscriminator(),
    )

    assert isinstance(module.training_config, SRGANTrainingConfig)
    assert isinstance(module.eval_config, SRGANEvalConfig)
    assert isinstance(module.adversarial_loss, AdversarialLoss)
    assert module.eval_config.ssim_impl == "daala"
    assert module.eval_config.perceptual_metrics == ["lpips", "dists"]
    assert module.automatic_optimization is False


def test_matching_discriminator_channels_are_accepted():
    """The check must key on the real channel counts, not refuse every pairing."""
    module = SRGANLightning(
        model=SRResNet(scale=4, in_out_channels=1, num_residual_blocks=1),
        processor=YChannelProcessor(),
        discriminator=SRDiscriminator(in_channels=1),
        training_config=SRGANTrainingConfig(),
    )

    assert module.discriminator.hparams["in_channels"] == 1


def test_ddp_refused():
    module = build_gan_module()
    module.trainer = SimpleNamespace(world_size=2, precision="32-true")

    with pytest.raises(RuntimeError, match="world_size"):
        module.on_fit_start()


def test_discriminator_input_size_checked_against_the_real_crop():
    """A D/crop mismatch is otherwise a raw Linear shape error deep in step 1."""
    module = build_gan_module(hr_input_size=128)  # dataset serves 96
    module.trainer = SimpleNamespace(datamodule=fake_datamodule(hr_crop_size=96))

    with pytest.raises(ValueError, match="hr_input_size"):
        module.setup("fit")


def test_a_matching_discriminator_input_size_passes_the_probe():
    """The probe must key on the real crop, not refuse every dataset."""
    module = build_gan_module(hr_input_size=96)
    module.trainer = SimpleNamespace(datamodule=fake_datamodule(hr_crop_size=96))

    module.setup("fit")  # must not raise


def test_the_probe_keys_on_the_hr_crop_the_discriminator_scores():
    """The discriminator sees extract_target(hr_cropped), not the raw HR patch.

    SRResNet emits exactly scale x lr, which makes the two indistinguishable, so
    a probe keyed on the raw patch looks correct. A shrinking pre-upsampled
    generator separates them: it takes a 112x112 patch and emits 96x96, and 96 is
    the size that reaches the discriminator's shape-fixed dense head.
    """
    module = SRGANLightning(
        model=ShrinkingGenerator(shrink=16),
        processor=RGBSignedOutputProcessor(),
        discriminator=SRDiscriminator(hr_input_size=96),
        training_config=SRGANTrainingConfig(example_input_shape=None),
    )
    module.trainer = SimpleNamespace(datamodule=fake_datamodule(hr_crop_size=112, scale=1))

    module.setup("fit")  # must not raise: the 112x112 patch is cropped to the 96x96 output


def test_the_size_check_is_scoped_to_the_train_dataset():
    """``hr_input_size`` constrains the TRAINING crop, and nothing else.

    ``setup`` probes whichever dataset the live stage instantiated, so a bare
    ``validate``/``test`` run hands the probe a **full image**. Ungated, that
    both refuses a perfectly valid run and does a full-resolution generator
    forward on CPU at setup.
    """
    module = build_gan_module(hr_input_size=96)
    module.trainer = SimpleNamespace(datamodule=val_only_datamodule(hr_size=128))

    module.setup("validate")  # must not raise: 128 != 96 is not a training crop


def test_the_probe_forward_runs_with_the_whole_module_in_eval_mode():
    """``self.model.eval()`` is not enough: the forward selector reads the
    LightningModule's own ``training`` flag (``self.training and self._compiled
    is not None``), which ``self.model.eval()`` leaves True. On a compiled run
    the probe would therefore go through ``self._compiled`` and trigger a dynamo
    compile at setup — before, and at a different shape from, the deliberate
    ``on_fit_start`` warm-up that exists to surface toolchain failures at a
    predictable point.

    The generator's own flag must stay False regardless: a train-mode probe
    forward folds the sample into its BatchNorm running statistics before
    training has taken a step.
    """
    module = build_gan_module(cls=ProbeStateSpy)
    module.trainer = SimpleNamespace(datamodule=fake_datamodule(hr_crop_size=96))

    module.setup("fit")

    assert module.probe_flags == [(False, False)], "probe forward ran in training mode"
    assert module.training and module.model.training, "training mode was not restored"


def test_layer_lrs_refused():
    """Inherited through SRResNetTrainingConfig, so YAML can set it — and this
    override never reads it, which would silently give both networks uniform LRs
    where the base raises. The file's only unsignalled misconfiguration."""
    module = build_gan_module(layer_lrs=[1e-4, 1e-4, 1e-4])

    with pytest.raises(ValueError, match="layer_lrs"):
        module.configure_optimizers()


def test_a_base_training_config_is_refused():
    """training_step reads adversarial_weight and d_steps_per_g_step off the
    config on every step, and only the subclass carries them — the type hint
    alone does not enforce it, so a base config would fail mid-run instead."""
    with pytest.raises(TypeError, match="SRGANTrainingConfig"):
        SRGANLightning(
            model=SRResNet(scale=4, num_residual_blocks=1),
            processor=RGBSignedOutputProcessor(),
            discriminator=SRDiscriminator(),
            training_config=SRTrainingConfig(),
        )


# ---------------------------------------------------------------------------
# init_from — MSE initialisation of the generator
# ---------------------------------------------------------------------------

#: Every meta field init_from validates, in the message form it reports them.
INIT_FROM_FIELDS = (
    "model.class_path",
    "model.init_args",
    "processor.class_path",
    "io.output_range",
    "io.scale",
)


def test_init_from_loads_generator_weights_bit_exactly(tmp_path):
    """Ledig scopes the MSE-init trick to "when training the actual GAN", so a
    paper-faithful run starts from an MSE-trained SRResNet rather than scratch.

    Bit-exactly, not approximately: the two modules are built independently, so
    an ignored ``init_from`` leaves two unrelated random initialisations here.
    """
    path, source = write_weights(tmp_path)

    gan = init_gan_from(path)

    loaded, original = gan.model.state_dict(), source.model.state_dict()
    assert loaded.keys() == original.keys()
    for key, value in original.items():
        assert torch.equal(loaded[key], value), key
    # Only the generator: the discriminator is not in that file at all, and the
    # adversarial half of the run starts from scratch by design.
    assert not any(k.startswith("discriminator.") for k in loaded)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"model.class_path": "sisr.models.srcnn.model.SRCNN"}, "model.class_path"),
        (
            {
                "model.init_args": {
                    "scale": 4,
                    "in_out_channels": 3,
                    "hidden_channel": 64,
                    "kernel_sizes": [9, 3, 9],
                    "num_residual_blocks": 16,
                    "padding": "same",
                }
            },
            "model.init_args",
        ),
        ({"processor.class_path": "sisr.processors.rgb.RGBProcessor"}, "processor.class_path"),
        ({"io.output_range": [0.0, 1.0]}, "io.output_range"),
        ({"io.scale": 2}, "io.scale"),
    ],
)
def test_init_from_refuses_a_mismatched_artifact(tmp_path, override, field):
    """Silently initialising from weights trained under a different processor,
    scale or architecture produces a model that trains and scores without ever
    erroring — only the numbers are wrong.

    Each field is refused on its own: a single generic "metadata mismatch" would
    satisfy a substring match on any one field name, so the message is asserted
    to name the offending field and *only* that one.
    """
    path, _ = write_weights(tmp_path, **override)

    with pytest.raises(ValueError) as excinfo:
        init_gan_from(path)

    message = str(excinfo.value)
    assert [name for name in INIT_FROM_FIELDS if name in message] == [field]


def test_init_from_accepts_an_artifact_whose_unvalidated_fields_differ(tmp_path):
    """The check must key on the five fields that make weights transferable, not
    refuse every artifact: a criterion or eval_config recorded by the source run
    says nothing about whether its weights fit this generator."""
    path, _ = write_weights(tmp_path, **{"eval_config.crop_border": 99})

    init_gan_from(path)  # must not raise


def test_init_from_rejects_a_ckpt_and_names_the_pt(tmp_path):
    """A ``.ckpt`` holds the whole module under ``model.``-prefixed keys, so a
    strict load into the bare generator fails with an unreadable key dump —
    and it carries no ``meta`` to validate against either."""
    path = tmp_path / "sr-42.ckpt"
    safetensors.torch.save_file({}, str(path))  # no provenance header at all

    with pytest.raises(ValueError) as excinfo:
        init_gan_from(path)

    message = str(excinfo.value)
    assert "sr-42.ckpt" in message
    assert "sr-weights" in message, "the message must point at the sibling weights file"


def test_init_from_refuses_a_pt_with_no_metadata(tmp_path):
    """Without ``meta`` there is nothing to validate against, so accepting the
    file would silently reinstate every mismatch the checks above refuse."""
    path = tmp_path / "sr-weights.safetensors"
    safetensors.torch.save_file(
        {k: v.contiguous() for k, v in build_source_module().model.state_dict().items()},
        str(path),
    )  # tensors, but no provenance header

    with pytest.raises(ValueError, match="no sisr provenance header"):
        init_gan_from(path)


def test_init_from_names_the_component_artifact_it_was_handed(tmp_path):
    """A ``d-weights-*`` file is the likeliest misuse after a ``.ckpt``: the template
    writes both files into one dirpath.

    It is refused either way — a component's meta has no generator-scoped ``io``
    section — but "io.scale is None" never tells the user they grabbed the
    discriminator's weights, which is the whole mistake.
    """
    module = build_gan_module()
    path = tmp_path / "d-weights-10000.safetensors"
    artifacts.save(
        path,
        module.discriminator.state_dict(),
        build_component_metadata(module, "discriminator"),
    )

    with pytest.raises(ValueError) as excinfo:
        init_gan_from(path)

    message = str(excinfo.value)
    assert "d-weights-10000.safetensors" in message
    assert "discriminator" in message, "the message must name the component it was handed"
    assert "sr-weights" in message, "...and the generator file that should have been used"
    assert not [name for name in INIT_FROM_FIELDS if name in message], (
        "a component file is not a field mismatch, and must not be reported as one"
    )


def test_init_from_accepts_a_pt_written_before_kind_existed(tmp_path):
    """``kind`` is an additive field: the golden 1e6-step artifact the template
    ships predates it, and keying the component refusal on its *absence* would
    refuse the one file ``init_from`` is shipped pointing at."""
    source = build_source_module()
    meta = build_metadata(source)
    del meta["kind"]
    path = tmp_path / "sr-weights.safetensors"
    artifacts.save(path, source.model.state_dict(), meta)

    init_gan_from(path)  # must not raise


def test_init_from_names_the_cli_null_coercion_trap():
    """``--...init_from=null`` does not unset it: jsonargparse coerces a null CLI
    value for a ``str | None`` field to the *string* ``'None'``, which then reaches
    ``torch.load`` as a filename and dies as a missing file. Name the trap instead —
    the comment in the template cannot reach someone who has already typed it."""
    with pytest.raises(ValueError) as excinfo:
        init_gan_from("None")

    message = str(excinfo.value)
    assert "null" in message, "the message must name the CLI override that produces it"
    assert "--config" in message, "...and the overlay that actually works"


def test_init_from_is_only_read_when_the_run_is_fitting(tmp_path):
    """Every subcommand instantiates the model from config — ``--ckpt_path`` does
    not skip it — so a construction-time load made ``validate``/``test``/``export``
    depend on an artifact they never use, and die with ``FileNotFoundError`` on
    every clone, worktree and CI box that lacks it.

    A path that does not exist is the whole proof: the non-fit stages must not
    touch it, and ``fit`` must.
    """
    missing = tmp_path / "absent-sr-weights.safetensors"

    module = build_gan_module(init_from=str(missing))  # construction must not read it
    for stage in ("validate", "test", "predict"):
        module.setup(stage)  # must not raise

    with pytest.raises(FileNotFoundError):
        module.setup("fit")


@_ignore_cpu_fit_warnings
def test_init_from_is_reached_through_a_real_trainer_fit(tmp_path):
    """The stage gate must hold for the value Lightning passes — ``trainer.state.fn``,
    a ``TrainerFn`` enum — not only the plain string a direct ``setup()`` call uses.

    A gate that missed it would skip the MSE initialisation of every real run
    without a word: the generator would just start from scratch, which is not the
    paper's recipe and looks like nothing at all. Driving a refusal out through
    ``trainer.fit`` is the cheap proof that the call site is reached.
    """
    path, _ = write_weights(tmp_path, **{"io.scale": 2})

    with pytest.raises(ValueError, match="io.scale"):
        fit_gan(build_gan_module(init_from=str(path)), n_batches=1)


# ---------------------------------------------------------------------------
# checkpoint provenance
# ---------------------------------------------------------------------------


@_ignore_cpu_fit_warnings
def test_checkpoint_carries_both_networks_and_their_optimizers(tmp_path):
    """The adversarial half of a run must be recoverable from its own ``.ckpt``.

    Driven by a real fit with checkpointing on, because everything asserted here
    is written by Lightning's save path rather than by any one method: the
    second optimizer's state, the discriminator's parameters, and the two
    ``sisr_meta`` blocks this module appends.

    The VGG criterion is what makes the final assertion non-vacuous — a
    perceptual criterion holds its frozen VGG outside the module tree precisely
    so up to 20M frozen parameters stay out of every checkpoint.
    """
    with pytest.warns(UserWarning, match="randomly initialised"):
        criterion = VGG19FeatureLoss(layer="vgg22", weights=None)
    module = build_gan_module(criterion=criterion)

    fit_gan(
        module,
        n_batches=2,
        enable_checkpointing=True,
        callbacks=[SRCheckpoint(monitor_metric=None, dirpath=str(tmp_path), every_n_train_steps=1)],
    )
    ckpt = torch.load(next(tmp_path.glob("*.ckpt")), weights_only=False)

    assert any(k.startswith("model.") for k in ckpt["state_dict"])
    assert any(k.startswith("discriminator.") for k in ckpt["state_dict"])
    assert len(ckpt["optimizer_states"]) == 2
    assert ckpt["sisr_meta"]["discriminator"]["class_path"].endswith("SRDiscriminator")
    assert ckpt["sisr_meta"]["discriminator"]["init_args"]["hr_input_size"] == 96
    assert ckpt["sisr_meta"]["adversarial"]["loss"].endswith("AdversarialLoss")
    assert ckpt["sisr_meta"]["adversarial"]["weight"] == pytest.approx(1e-3)
    assert ckpt["sisr_meta"]["adversarial"]["d_steps_per_g_step"] == 1
    # The base's generator-scoped provenance survives — the generator is still
    # the distributable artifact.
    assert ckpt["sisr_meta"]["model"]["class_path"].endswith("SRResNet")
    # Structural, not a name match: `not any("vgg" in k)` would stay green if the
    # criterion's frozen backbone were ever renamed, and 20M parameters would land
    # in every checkpoint unnoticed.
    assert all(k.startswith(("model.", "discriminator.")) for k in ckpt["state_dict"])


@_ignore_cpu_fit_warnings
def test_the_bare_weights_sink_stays_component_scoped(tmp_path):
    """``on_save_checkpoint`` is a ``.ckpt`` hook only.

    ``SRWeightsCheckpoint`` writes one network's weights and describes exactly
    that network; adding this run's whole adversarial setup to a discriminator's
    (or a generator's) ``.pt`` would describe things the file does not contain.
    Both sinks are checked — the template runs one of each side by side, and the
    generator's is the distributable artifact ``init_from`` consumes.

    Driven by a real fit, and asserted on the files that fit wrote: the property is
    about which metadata builder the save path reaches, so inspecting the builders
    directly cannot show it — neither has ever carried these keys.
    """
    module = build_gan_module()

    fit_gan(
        module,
        n_batches=2,
        enable_checkpointing=True,
        callbacks=[
            SRWeightsCheckpoint(
                monitor_metric=None,
                dirpath=str(tmp_path),
                every_n_train_steps=1,
                attribute="discriminator",
                filename_prefix="d-weights",
            ),
            SRWeightsCheckpoint(
                monitor_metric=None,
                dirpath=str(tmp_path),
                every_n_train_steps=1,
                attribute="model",
                filename_prefix="sr-weights",
            ),
        ],
    )
    d_meta = artifacts.load(next(tmp_path.glob("d-weights_s*.safetensors")))[1]
    g_meta = artifacts.load(next(tmp_path.glob("sr-weights_s*.safetensors")))[1]

    assert set(d_meta).isdisjoint({"discriminator", "adversarial"})
    assert d_meta["kind"] == "component"
    assert d_meta["component"]["name"] == "discriminator"
    assert set(g_meta).isdisjoint({"discriminator", "adversarial"})
    assert g_meta["kind"] == "sr_model"
