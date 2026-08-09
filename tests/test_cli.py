"""CLI tests: in-process config resolution.

Every assertion resolves a config via ``SRLightningCLI(args=..., run=False)`` (see
``_resolve``) or via a synchronous ``--help`` invocation — nothing spawns a
subprocess, so nothing pays the torch + lightning cold-import cost per test. The
installed ``sisr`` console script itself is smoke-tested in ``build.yml`` instead
(a packaging concern, not something every test run needs to re-pay for).
"""

import sys
import warnings
from pathlib import Path

import lightning
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "templates" / "config.srcnn.template.yaml"
SRRESNET_TEMPLATE = REPO_ROOT / "templates" / "config.srresnet.template.yaml"
SRGAN_TEMPLATE = REPO_ROOT / "templates" / "config.srgan.template.yaml"

_ignore_random_vgg = pytest.mark.filterwarnings("ignore:.*randomly initialised.*:UserWarning")


def _offline_args(template_path: Path, tmp_path: Path) -> list[str]:
    """Extra ``--config`` overlay a template needs to resolve off a training box.

    Only the SRGAN template needs one; every other template resolves as shipped.
    Written as an overlay rather than ``--key=null`` because jsonargparse coerces
    a ``null`` CLI value for a ``str | None`` field to the *string* ``'None'``,
    which is exactly the silent-wrong-value class these two settings guard.

    ``init_from`` points at a golden ``.pt`` under the gitignored ``experiments/``
    tree — present on a training box, absent in CI and in every worktree.
    ``criterion.weights`` would otherwise download ~548 MB of VGG19 weights just
    to resolve a config; ``null`` builds a random VGG (and warns).
    """
    if template_path.name != SRGAN_TEMPLATE.name:
        return []
    overlay = tmp_path / "srgan_offline_overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "init_args": {
                        "training_config": {"init_args": {"init_from": None}},
                        "criterion": {
                            "class_path": "sisr.losses.VGG19FeatureLoss",
                            "init_args": {"layer": "vgg54", "weights": None},
                        },
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return ["--config", str(overlay)]


SUBCLASS_MODE_CONFIG = """
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 1.0e-4
model:
  class_path: sisr.training.SRLightning
  init_args:
    model:
      class_path: sisr.models.srresnet.SRResNet
      init_args:
        scale: 4
    processor:
      class_path: sisr.processors.RGBSignedOutputProcessor
    training_config:
      class_path: sisr.models.srresnet.SRResNetTrainingConfig
      init_args:
        example_input_shape: [3, 24, 24]
    eval_config:
      class_path: sisr.models.srresnet.SRResNetEvalConfig
data:
  train_dataset:
    class_path: sisr.datasets.srresnet.TrainDataset
    init_args:
      img_dir: IMG_DIR
      scale: 4
      hr_crop_size: 96
  val_dataset:
    class_path: sisr.datasets.srresnet.ValidationDataset
    init_args:
      img_dir: IMG_DIR
      scale: 4
trainer:
  accelerator: cpu
  devices: 1
  logger: false
  enable_checkpointing: false
"""


def _resolve(*args: str):
    """Resolve a config fully in-process via ``SRLightningCLI(run=False)``.

    No subprocess, no CUDA: jsonargparse merges the YAML + CLI overrides +
    dataclass defaults and instantiates the classes, but never runs a training
    loop. ``--trainer.accelerator=cpu --trainer.devices=1`` is appended because
    ``run=False`` instantiates the Trainer, and both shipped templates request
    CUDA — the override lets resolution succeed on any machine without touching
    the model/processor/config fields the tests assert on.
    """
    from sisr.cli import SRLightningCLI
    from sisr.training import SRDataModule, SRLightning

    # LightningCLI warns when both `args=` and sys.argv[1:] are set (it sees pytest's
    # own argv); the strict global filterwarnings=error would turn that into a failure.
    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        # Two framework-internal warnings only fire on this in-process path and would
        # trip the strict "error" filter, so suppress them locally: jsonargparse's
        # DeprecationWarning (lightning's deprecated add_instantiator call), and "GPU
        # available but not used" (run=False + forced accelerator=cpu, harmless).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message="GPU available but not used.*")
            return SRLightningCLI(
                model_class=SRLightning,
                datamodule_class=SRDataModule,
                subclass_mode_model=True,
                auto_configure_optimizers=False,
                save_config_kwargs={"overwrite": True},
                args=[*args, "--trainer.accelerator=cpu", "--trainer.devices=1"],
                run=False,
            )
    finally:
        sys.argv = saved_argv


def test_subclass_mode_accepts_a_class_path_model_block(tmp_path):
    """A YAML naming its Lightning module by class_path builds that module.

    Guards the whole point of subclass mode: before it, model_class was fixed
    and no config could select a different module.
    """
    from sisr.cli import SRLightningCLI
    from sisr.training import SRDataModule, SRLightning

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        SUBCLASS_MODE_CONFIG.replace("IMG_DIR", str(tmp_path).replace("\\", "/")),
        encoding="utf-8",
    )
    # See _resolve's comment: LightningCLI warns when both args= and pytest's own
    # sys.argv are set, which the strict global filterwarnings=error would fail on.
    # SUBCLASS_MODE_CONFIG also carries no seed_everything: (unlike the shipped
    # templates), so Lightning's own "No seed found" UserWarning needs the same
    # local suppression.
    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message="GPU available but not used.*")
            warnings.filterwarnings("ignore", message="No seed found.*")
            cli = SRLightningCLI(
                model_class=SRLightning,
                datamodule_class=SRDataModule,
                subclass_mode_model=True,
                auto_configure_optimizers=False,
                run=False,
                args=["--config", str(cfg)],
            )
    finally:
        sys.argv = saved_argv
    assert isinstance(cli.model, SRLightning)
    optimizer = cli.model.optimizer([torch.zeros(1, requires_grad=True)])
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_srcnn_config_resolves_in_process():
    """SRCNN template resolves to the paper-faithful model/processor/config
    classes — resolved in-process (no subprocess).

    Ports the class-wiring assertions of the old subprocess ``--print_config``
    test. ``cli.config`` is the same merged config ``--print_config`` dumps, so
    these assert on resolved objects rather than on dumped YAML text.
    """
    from sisr.models.srcnn import SRCNN, SRCNNEvalConfig, SRCNNTrainingConfig
    from sisr.processors import YChannelProcessor

    cli = _resolve("--config", str(TEMPLATE))
    m = cli.model
    assert isinstance(m.model, SRCNN)
    assert isinstance(m.processor, YChannelProcessor)
    assert isinstance(m.training_config, SRCNNTrainingConfig)
    assert isinstance(m.eval_config, SRCNNEvalConfig)
    assert m.eval_config.crop_border == 3  # inherited-default check
    assert m.eval_config.ssim_impl == "wang"  # inherited-default check
    # Paper recipe: reconstruction layer learns 10x slower than the other two.
    assert m.training_config.layer_lrs == [1.0e-4, 1.0e-4, 1.0e-5]
    # Top-level optimizer block linked from YAML.
    assert cli.config.optimizer.class_path == "torch.optim.SGD"
    # The removed model_colorspace field must not reappear.
    assert not hasattr(m, "model_colorspace")
    assert not hasattr(m.eval_config, "model_colorspace")


def test_srresnet_config_resolves_in_process():
    """SRResNet template resolves to the paper-faithful model/processor/config
    classes + the random-crop dataset specs — resolved in-process (no subprocess).
    Ports the class-wiring assertions of the old subprocess --print_config test."""
    from sisr.models.srresnet import (
        SRResNet,
        SRResNetEvalConfig,
        SRResNetTrainingConfig,
    )
    from sisr.processors import RGBSignedOutputProcessor

    cli = _resolve("--config", str(SRRESNET_TEMPLATE))
    m = cli.model
    assert isinstance(m.model, SRResNet)
    # The template ships the paper's [-1, 1] HR/target range.
    assert isinstance(m.processor, RGBSignedOutputProcessor)
    assert isinstance(m.training_config, SRResNetTrainingConfig)
    assert isinstance(m.eval_config, SRResNetEvalConfig)
    assert m.eval_config.crop_border == 4  # inherited-default check
    assert m.eval_config.ssim_impl == "daala"  # inherited-default check
    # Dataset specs stay plain {class_path, init_args} dicts (materialized lazily
    # in SRDataModule.setup), so assert on the resolved raw config.
    assert cli.config.data.train_dataset["class_path"] == "sisr.datasets.srresnet.TrainDataset"
    assert cli.config.data.val_dataset["class_path"] == "sisr.datasets.srresnet.ValidationDataset"
    assert cli.config.data.train_dataset["init_args"]["hr_crop_size"] == 96
    # The removed model_colorspace field must not reappear anywhere.
    assert not hasattr(m, "model_colorspace")
    assert not hasattr(m.eval_config, "model_colorspace")


@_ignore_random_vgg
def test_srgan_template_parses_and_builds(tmp_path: Path):
    """The YAML path is the whole point; constructing the module in Python proves
    nothing about it. A prior arc shipped a composite-loss feature whose only
    tests built it in Python, leaving the YAML wiring — the deliverable —
    unguarded.

    The nested discriminator/discriminator_optimizer blocks are the part most
    likely to break: they are nested under ``model.init_args`` rather than
    top-level precisely because a second ``add_optimizer_args(link_to=...)``
    would target an argument ``SRLightning`` does not have, so nothing else in
    the repo exercises that shape.
    """
    from sisr.losses import AdversarialLoss, VGG19FeatureLoss
    from sisr.models.srgan import SRDiscriminator, SRGANEvalConfig, SRGANTrainingConfig
    from sisr.models.srresnet import SRResNet
    from sisr.processors import RGBSignedOutputProcessor
    from sisr.training import SRGANLightning

    cli = _resolve("--config", str(SRGAN_TEMPLATE), *_offline_args(SRGAN_TEMPLATE, tmp_path))

    m = cli.model
    assert isinstance(m, SRGANLightning)
    assert isinstance(m.model, SRResNet)
    assert isinstance(m.processor, RGBSignedOutputProcessor)
    assert isinstance(m.discriminator, SRDiscriminator)
    assert isinstance(m.adversarial_loss, AdversarialLoss)
    assert isinstance(m.training_config, SRGANTrainingConfig)
    assert isinstance(m.eval_config, SRGANEvalConfig)
    # Ledig's phi_5,4 content loss, with no total-variation term: TV at 2e-8 is
    # scoped in the paper to SRResNet-VGG22, not to the GAN variants.
    assert isinstance(m.criterion, VGG19FeatureLoss)
    assert m.criterion.layer == "vgg54"
    assert m.training_config.adversarial_weight == pytest.approx(1e-3)
    # The discriminator's dense head fixes its input size, so the train crop has
    # to agree with it or the first step is a raw Linear shape error.
    assert m.discriminator.hparams["hr_input_size"] == 96
    assert cli.config.data.train_dataset["init_args"]["hr_crop_size"] == 96
    # The nested optimizer block really produces a discriminator optimizer —
    # OptimizerCallable resolution off a nested key, not off a linked top-level one.
    opt_d = m.discriminator_optimizer(m.discriminator.parameters())
    assert isinstance(opt_d, torch.optim.Adam)
    assert opt_d.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert {id(p) for group in opt_d.param_groups for p in group["params"]} == {
        id(p) for p in m.discriminator.parameters()
    }
    # ...and the top-level lr_scheduler: key still links into the GAN module under
    # subclass mode, which is what carries the paper's second training phase.
    opt_g = m.optimizer(m.model.parameters())
    assert isinstance(m.lr_scheduler(opt_g), torch.optim.lr_scheduler.MultiStepLR)
    # ...and the nested discriminator_lr_scheduler decays the critic in step with
    # the generator — Ledig et al.'s one schedule describes the SRGAN networks as
    # a whole, not the generator alone.
    assert isinstance(m.discriminator_lr_scheduler(opt_d), torch.optim.lr_scheduler.MultiStepLR)


def test_srgan_template_ships_the_papers_full_two_phase_schedule():
    """Ledig trains 1e5 generator iterations at 1e-4, then 1e5 more at 1e-5.

    This module takes two optimizer steps per batch, so 2e5 iterations is
    ``max_steps: 400000`` *global steps*, while the decay milestone is in BATCHES —
    ``SRGANLightning`` steps its schedulers by hand, once per batch. Shipping half
    the schedule, or the full length with the scheduler commented out, reproduces
    neither phase of the paper.

    ``val_check_interval`` is in batches too (Lightning keys it on
    ``total_batch_idx`` whenever ``check_val_every_n_epoch`` is null, which this
    template sets), so 5000 there is the same 10000-global-step cadence the
    checkpoint callbacks fire on.

    The discriminator must decay on the same schedule as the generator: Ledig et
    al.'s sentence describes the SRGAN networks as a whole, and a critic left at
    1e-4 for the whole second phase while the generator drops to 1e-5 is a
    standard way to saturate the generator's adversarial gradient.
    """
    with SRGAN_TEMPLATE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["trainer"]["max_steps"] == 400000
    assert data["trainer"]["check_val_every_n_epoch"] is None
    assert data["trainer"]["val_check_interval"] == 5000
    assert data["lr_scheduler"]["class_path"].endswith("MultiStepLR")
    assert data["lr_scheduler"]["init_args"]["milestones"] == [100000]
    assert data["lr_scheduler"]["init_args"]["gamma"] == pytest.approx(0.1)
    d_scheduler = data["model"]["init_args"]["discriminator_lr_scheduler"]
    assert d_scheduler["class_path"].endswith("MultiStepLR")
    assert d_scheduler["init_args"]["milestones"] == [100000]
    assert d_scheduler["init_args"]["gamma"] == pytest.approx(0.1)


@_ignore_random_vgg
def test_srgan_template_resolves_without_the_init_from_artifact(tmp_path: Path):
    """Instantiating the model is what every subcommand does — ``validate``/``test``/
    ``export --ckpt_path`` included — so reading ``init_from`` at construction made
    all of them depend on the gitignored golden ``.pt``. This resolves the shipped
    template with ``init_from`` pointed at a file that does not exist: it must build,
    because nothing here is fitting.
    """
    overlay = tmp_path / "missing_init_from.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "init_args": {
                        "training_config": {
                            "init_args": {"init_from": str(tmp_path / "absent.pt")}
                        },
                        "criterion": {
                            "class_path": "sisr.losses.VGG19FeatureLoss",
                            "init_args": {"layer": "vgg54", "weights": None},
                        },
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cli = _resolve("--config", str(SRGAN_TEMPLATE), "--config", str(overlay))

    assert cli.model.training_config.init_from == str(tmp_path / "absent.pt")


def test_srgan_template_ships_a_real_init_from_path():
    """The template's own init_from is exercised nowhere else — the test above
    overrides it to a missing path so it can run without the gitignored
    experiments/ tree. Assert the shipped value at least names a bare-weights
    .pt, which is one of the artifacts SRGANLightning's setup refuses."""
    with SRGAN_TEMPLATE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    init_from = data["model"]["init_args"]["training_config"]["init_args"]["init_from"]
    assert init_from.endswith(".pt")
    assert "sr-weights" in init_from


def test_composite_criterion_resolves_through_the_real_cli(tmp_path: Path):
    """The paper's SRResNet-VGG22 recipe (Ledig et al. §3.4: a VGG22 content
    term plus total variation at 2e-8, no pixel term) resolved from YAML through
    the real CLI, not constructed in Python.

    WeightedSumLoss(terms: dict[str, torch.nn.Module]) with per-term nested
    class_path/init_args relies on one specific jsonargparse capability that
    every other composite test bypasses by building objects directly. This is
    the one test that would catch a jsonargparse regression, a typo'd
    class_path in the README/templates, or a terms/weights rename.

    weights: null on the VGG term keeps this offline (random-init VGG) and
    fires the "randomly initialised" UserWarning at instantiation time, which
    the strict filterwarnings=error config would otherwise turn into a failure.
    """
    from sisr.losses import TotalVariationLoss, VGG19FeatureLoss, WeightedSumLoss

    overlay = {
        "model": {
            "criterion": {
                "class_path": "sisr.losses.WeightedSumLoss",
                "init_args": {
                    "terms": {
                        "vgg22": {
                            "class_path": "sisr.losses.VGG19FeatureLoss",
                            "init_args": {"layer": "vgg22", "weights": None},
                        },
                        "tv": {"class_path": "sisr.losses.TotalVariationLoss"},
                    },
                    "weights": {"vgg22": 1.0, "tv": 2.0e-8},
                },
            }
        }
    }
    overlay_path = tmp_path / "criterion_overlay.yaml"
    # sort_keys=False: yaml.safe_dump's default alphabetical sort would reorder
    # "terms" (tv before vgg22), and describe()'s order follows insertion order.
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False))

    with pytest.warns(UserWarning, match="randomly initialised"):
        cli = _resolve("--config", str(SRRESNET_TEMPLATE), "--config", str(overlay_path))

    criterion = cli.model.criterion
    assert isinstance(criterion, WeightedSumLoss)
    assert isinstance(criterion.terms["vgg22"], VGG19FeatureLoss)
    assert isinstance(criterion.terms["tv"], TotalVariationLoss)
    assert criterion.weights == {"vgg22": pytest.approx(1.0), "tv": pytest.approx(2.0e-8)}
    assert criterion.describe() == "1*vgg22 + 2e-08*tv"


def test_test_subcommand_help_exposes_ckpt_path_in_process(capsys, monkeypatch):
    """`test --help` documents --ckpt_path and --data.test_datasets.

    run=True + args=['test','--help'] parses in-process and exits after printing
    help (SystemExit) — no cold subprocess, no class instantiation. sys.argv is
    blanked so LightningCLI doesn't warn (→ error under the strict filter) about
    seeing both args= and pytest's own argv."""
    from sisr.cli import SRLightningCLI
    from sisr.training import SRDataModule, SRLightning

    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    with pytest.raises(SystemExit):
        SRLightningCLI(
            model_class=SRLightning,
            datamodule_class=SRDataModule,
            subclass_mode_model=True,
            auto_configure_optimizers=False,
            save_config_kwargs={"overwrite": True},
            args=["test", "--help"],
            run=True,
        )
    out = capsys.readouterr().out
    assert "--ckpt_path" in out
    assert "--data.test_datasets" in out


def test_export_subcommand_help_exposes_its_args_in_process(capsys, monkeypatch):
    """`export --help` documents --output_path, --ckpt_path, --opset_version.

    trainer_class is left at SRLightningCLI's default (_ExportTrainer) —
    building this parser never calls sisr.export.to_onnx's body (only inspects
    _ExportTrainer.export's signature), so it needs no onnx/onnxruntime install.
    """
    from sisr.cli import SRLightningCLI
    from sisr.training import SRDataModule, SRLightning

    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    with pytest.raises(SystemExit):
        SRLightningCLI(
            model_class=SRLightning,
            datamodule_class=SRDataModule,
            subclass_mode_model=True,
            auto_configure_optimizers=False,
            save_config_kwargs={"overwrite": True},
            args=["export", "--help"],
            run=True,
        )
    out = capsys.readouterr().out
    assert "--output_path" in out
    assert "--ckpt_path" in out
    assert "--opset_version" in out


def test_export_subcommand_runs_end_to_end_in_process(tmp_path):
    """`sisr export --config ... --output_path ...` produces a real ONNX file.

    Exercises the full subcommand wiring (_ExportTrainer.export -> to_onnx)
    against the shipped SRCNN template, whose dataset dirs need not exist —
    export never calls SRDataModule.setup(). Skips cleanly without the
    optional onnx/onnxruntime extra.
    """
    pytest.importorskip("onnx")
    from sisr.cli import SRLightningCLI
    from sisr.training import SRDataModule, SRLightning

    output_path = tmp_path / "srcnn.onnx"
    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message="GPU available but not used.*")
            SRLightningCLI(
                model_class=SRLightning,
                datamodule_class=SRDataModule,
                subclass_mode_model=True,
                auto_configure_optimizers=False,
                save_config_kwargs={"overwrite": True},
                args=[
                    "export",
                    "--config",
                    str(TEMPLATE),
                    "--output_path",
                    str(output_path),
                    "--trainer.accelerator=cpu",
                    "--trainer.devices=1",
                ],
            )
    finally:
        sys.argv = saved_argv

    assert output_path.is_file()


def test_optimizer_lr_override_in_process():
    """Top-level --optimizer.init_args.lr override surfaces in the resolved config."""
    cli = _resolve("--config", str(TEMPLATE), "--optimizer.init_args.lr=5e-3")
    assert cli.config.optimizer.init_args.lr == pytest.approx(0.005)


def test_dataset_cli_override_fails_loudly_instead_of_silently_ignored():
    """--data.train_dataset.init_args.crops_per_image=8 must not silently no-op.

    jsonargparse treats `data.train_dataset` as an opaque `dict[str, Any]`, so a
    dotted CLI override targeting a nested `init_args` key lands as a stray
    sibling key next to `class_path`/`init_args` instead — `instantiate_class`
    never sees it, and the constructed dataset silently keeps
    crops_per_image=1. SRDataModule must reject the malformed spec loudly at
    construction (during `instantiate_classes()`, i.e. CLI resolution here)
    instead of quietly building the wrong dataset.
    """
    with pytest.raises(ValueError, match="crops_per_image"):
        _resolve(
            "--config",
            str(SRRESNET_TEMPLATE),
            "--data.train_dataset.init_args.crops_per_image=8",
        )


def test_dataset_whole_dict_cli_override_colliding_with_config_fails_loudly():
    """--data.train_dataset='{...}' colliding with the template's own
    train_dataset (--config already sets one) must not crash with
    jsonargparse's raw, unactionable AttributeError ('dict' object has no
    attribute 'init_args'). SRLightningCLI.parse_arguments must convert that
    into an actionable SystemExit instead.
    """
    with pytest.raises(SystemExit, match="already has a class_path"):
        _resolve(
            "--config",
            str(SRRESNET_TEMPLATE),
            '--data.train_dataset={"class_path": "sisr.datasets.srresnet.TrainDataset", '
            '"init_args": {"img_dir": "data/DIV2K_train_HR", "scale": 4, "hr_crop_size": 96, '
            '"crops_per_image": 8, "use_tqdm": true, "cache_dir": ".lmdb_cache/DIV2K_train_HR"}}',
        )


def test_dataset_whole_init_args_cli_override_colliding_with_config_fails_loudly():
    """--data.train_dataset.init_args='{...}' hits the same jsonargparse crash
    (a whole-dict CLI value merged against an already-dict prev_val the
    template set) and must also fail loudly with an actionable SystemExit."""
    with pytest.raises(SystemExit, match="already has a class_path"):
        _resolve(
            "--config",
            str(SRRESNET_TEMPLATE),
            '--data.train_dataset.init_args={"crops_per_image": 8}',
        )


def test_dataset_whole_dict_override_without_prior_config_value_still_works():
    """Regression: --data.predict_dataset='{...}' must keep working when
    neither shipped template sets predict_dataset — there is no prior dict
    value to collide with, so jsonargparse's merge never hits the
    prev_val.init_args crash. This is the documented `sisr predict` workflow
    (SRDataModule.predict_dataloader's own error message tells users to set
    data.predict_dataset exactly this way). A blanket pre-scan that rejects
    every whole-dict dataset override regardless of a prior value would break
    this working case — the guard must only intercept an actual collision.
    """
    cli = _resolve(
        "--config",
        str(SRRESNET_TEMPLATE),
        '--data.predict_dataset={"class_path": "sisr.datasets.predict.PredictDataset", '
        '"init_args": {"img_dir": "data/Set5_HR"}}',
    )
    assert cli.config.data.predict_dataset["class_path"] == "sisr.datasets.predict.PredictDataset"
    assert cli.config.data.predict_dataset["init_args"]["img_dir"] == "data/Set5_HR"


def test_matmul_precision_accepted_in_process():
    """--matmul_precision=medium overrides the template's shipped 'high' default."""
    cli = _resolve("--config", str(TEMPLATE), "--matmul_precision=medium")
    assert cli.config.matmul_precision == "medium"


def test_matmul_precision_template_default_is_high_in_process():
    """The shipped templates set matmul_precision: high (TF32 on Ampere+) by default."""
    cli = _resolve("--config", str(TEMPLATE))
    assert cli.config.matmul_precision == "high"


def test_matmul_precision_rejects_invalid_in_process():
    """Invalid matmul_precision is rejected by the Literal validator during parse
    (SystemExit from jsonargparse), before any class is instantiated."""
    with pytest.raises(SystemExit):
        _resolve("--config", str(TEMPLATE), "--matmul_precision=bogus")


def test_trainer_benchmark_template_default_is_true_in_process():
    """cuDNN autotuning rides on Lightning's own trainer.benchmark, not a custom key.

    Trainer(benchmark=...) already assigns torch.backends.cudnn.benchmark and
    coordinates it with trainer.deterministic; a separate top-level key would
    bypass that coordination.
    """
    cli = _resolve("--config", str(TEMPLATE))
    assert cli.config.trainer.benchmark is True
    assert not hasattr(cli.config, "cudnn_benchmark")


# In-process unit tests for SRLightningCLI.before_instantiate_classes. Subprocess
# tests above cover argparse wiring; these cover the hook's branching logic so it
# shows up in line coverage.
def _make_cli_stub(subcommand: str | None, matmul_precision: str | None):
    """Build an SRLightningCLI instance bypassing __init__ for direct hook testing."""
    from types import SimpleNamespace

    from sisr.cli import SRLightningCLI

    cli = SRLightningCLI.__new__(SRLightningCLI)
    cli.subcommand = subcommand
    cli.config = (
        {subcommand: SimpleNamespace(matmul_precision=matmul_precision)} if subcommand else {}
    )
    return cli


def test_before_instantiate_classes_calls_torch_setter(monkeypatch):
    """When matmul_precision is set, torch.set_float32_matmul_precision is called with it."""
    import torch

    calls: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda p: calls.append(p))

    _make_cli_stub(subcommand="fit", matmul_precision="medium").before_instantiate_classes()

    assert calls == ["medium"]


def test_before_instantiate_classes_skips_when_unset(monkeypatch):
    """When matmul_precision is None, torch.set_float32_matmul_precision is NOT called."""
    import torch

    calls: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda p: calls.append(p))

    _make_cli_stub(subcommand="fit", matmul_precision=None).before_instantiate_classes()

    assert calls == []


def test_before_instantiate_classes_skips_when_no_subcommand(monkeypatch):
    """The hook returns early when self.subcommand is None (e.g., --help)."""
    import torch

    calls: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda p: calls.append(p))

    _make_cli_stub(subcommand=None, matmul_precision="medium").before_instantiate_classes()

    assert calls == []


TEMPLATE_PATHS = sorted((REPO_ROOT / "templates").glob("config.*.template.yaml"))
assert TEMPLATE_PATHS, "No templates found — check REPO_ROOT"


@pytest.mark.parametrize("template_path", TEMPLATE_PATHS, ids=[p.name for p in TEMPLATE_PATHS])
def test_template_yaml_parses(template_path: Path):
    """Every template YAML file must be valid YAML and have required keys."""
    with template_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{template_path} did not parse as a mapping"
    assert "trainer" in data and "model" in data and "data" in data


@_ignore_random_vgg
@pytest.mark.parametrize("template_path", TEMPLATE_PATHS, ids=[p.name for p in TEMPLATE_PATHS])
def test_template_disables_default_hp_metric(template_path: Path, tmp_path: Path):
    """Each shipped template must disable TensorBoard's hp_metric: -1 placeholder.

    Resolved in-process: cli.config is the same merged config --print_config would
    dump, so an inherited/overridden default is honored (checking the raw YAML
    would miss that)."""
    cli = _resolve("--config", str(template_path), *_offline_args(template_path, tmp_path))
    loggers = cli.config.trainer.logger
    tb = next(logger for logger in loggers if str(logger.class_path).endswith("TensorBoardLogger"))
    assert tb.init_args.default_hp_metric is False, (
        f"{template_path.name} does not disable the hp_metric placeholder. "
        f"Add `default_hp_metric: false` under the TensorBoardLogger init_args."
    )


# ---------------------------------------------------------------------------
# --ckpt_path round trip (real checkpoint, not fast_dev_run — which disables
# checkpointing entirely and so never exercises this)
# ---------------------------------------------------------------------------


def _run_cli(args: list[str]):
    """Run SRLightningCLI in-process for the given args, restoring sys.argv after.

    Mirrors ``_resolve``'s argv save/restore, but with ``run=True`` (the
    default) so the subcommand actually executes — needed here since the
    checkpoint round trip requires a real ``fit`` and a real second parse of
    ``--ckpt_path`` through ``LightningCLI._parse_ckpt_path``, not just config
    resolution. Returns the constructed ``SRLightningCLI`` so a caller can
    inspect ``.model`` as resolved *after* any ``--ckpt_path`` hyperparameter
    merge — ``_resolve``'s ``run=False`` path can't observe that merge at all,
    since ``_parse_ckpt_path`` no-ops when no subcommand was set up.
    """
    from sisr.cli import SRLightningCLI
    from sisr.training import SRDataModule, SRLightning

    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message="GPU available but not used.*")
            warnings.filterwarnings(
                "ignore", category=lightning.pytorch.utilities.warnings.PossibleUserWarning
            )
            return SRLightningCLI(
                model_class=SRLightning,
                datamodule_class=SRDataModule,
                subclass_mode_model=True,
                auto_configure_optimizers=False,
                save_config_callback=None,
                args=args,
            )
    finally:
        sys.argv = saved_argv


def _build_srcnn_checkpoint(tiny_rgb_image_dir: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Run a real (non-``fast_dev_run``) one-step ``fit`` and return (config_path, ckpt_path).

    ``fast_dev_run`` disables checkpointing entirely, so a real ``ModelCheckpoint``
    with ``every_n_train_steps=1`` is wired in instead to guarantee a ``.ckpt`` file
    after exactly one training step over the tiny fixture images.
    """
    ckpt_dir = tmp_path / "checkpoints"
    config = {
        "model": {
            "class_path": "sisr.training.SRLightning",
            "init_args": {
                "model": {
                    "class_path": "sisr.models.srcnn.SRCNN",
                    "init_args": {
                        "num_channels": 3,
                        "num_filters": [64, 32],
                        "kernel_sizes": [9, 1, 5],
                        "padding": 0,
                    },
                },
                "processor": {"class_path": "sisr.processors.RGBProcessor"},
                "eval_config": {
                    "class_path": "sisr.training.SREvalConfig",
                    "init_args": {"crop_border": 0},
                },
            },
        },
        "optimizer": {"class_path": "torch.optim.SGD", "init_args": {"lr": 1.0e-4}},
        "data": {
            "train_dataset": {
                "class_path": "sisr.datasets.srcnn.TrainDataset",
                "init_args": {
                    "img_dir": str(tiny_rgb_image_dir),
                    "subimg_size": 33,
                    "stride": 14,
                    "scale": 2,
                    "use_tqdm": False,
                    "cache_dir": str(tmp_path / ".lmdb_cache"),
                },
            },
            "val_dataset": {
                "class_path": "sisr.datasets.srcnn.ValidationDataset",
                "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
            },
            "train_dataloader_kwargs": {"batch_size": 2, "num_workers": 0},
            "val_dataloader_kwargs": {"batch_size": 1, "num_workers": 0},
        },
        "trainer": {
            "max_epochs": 1,
            "max_steps": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "default_root_dir": str(tmp_path),
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "dirpath": str(ckpt_dir),
                        "filename": "sr-test",
                        "every_n_train_steps": 1,
                        "save_top_k": -1,
                    },
                }
            ],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    _run_cli(["fit", "--config", str(config_path)])

    ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
    assert ckpt_files, "fit did not write a checkpoint"
    return config_path, ckpt_files[-1]


def _build_srresnet_checkpoint(tiny_rgb_image_dir: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Run a real one-step SRResNet fit and return (config_path, ckpt_path).

    Mirrors ``_build_srcnn_checkpoint``, but over SRResNet's native-LR pipeline
    and its ``SRResNetEvalConfig`` — the only shipped eval_config subclass with
    a subclass-only default (``ssim_impl='daala'``), which is what
    ``test_ckpt_path_preserves_eval_config_subclass_identity`` needs to observe.
    """
    ckpt_dir = tmp_path / "checkpoints"
    config = {
        "model": {
            "class_path": "sisr.training.SRLightning",
            "init_args": {
                "model": {
                    "class_path": "sisr.models.srresnet.SRResNet",
                    "init_args": {
                        "scale": 2,
                        "in_out_channels": 3,
                        "hidden_channel": 4,
                        "kernel_sizes": [3, 3, 3],
                        "num_residual_blocks": 1,
                        "padding": "same",
                    },
                },
                "processor": {"class_path": "sisr.processors.RGBProcessor"},
                "training_config": {
                    "class_path": "sisr.models.srresnet.SRResNetTrainingConfig",
                    "init_args": {"scale": 2},
                },
                "eval_config": {"class_path": "sisr.models.srresnet.SRResNetEvalConfig"},
            },
        },
        "optimizer": {"class_path": "torch.optim.SGD", "init_args": {"lr": 1.0e-4}},
        "data": {
            "train_dataset": {
                "class_path": "sisr.datasets.srresnet.TrainDataset",
                "init_args": {
                    "img_dir": str(tiny_rgb_image_dir),
                    "scale": 2,
                    "hr_crop_size": 24,
                    "use_tqdm": False,
                    "cache_dir": str(tmp_path / ".lmdb_cache"),
                },
            },
            "val_dataset": {
                "class_path": "sisr.datasets.srresnet.ValidationDataset",
                "init_args": {"img_dir": str(tiny_rgb_image_dir), "scale": 2},
            },
            "train_dataloader_kwargs": {"batch_size": 2, "num_workers": 0},
            "val_dataloader_kwargs": {"batch_size": 1, "num_workers": 0},
        },
        "trainer": {
            "max_epochs": 1,
            "max_steps": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "default_root_dir": str(tmp_path),
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "dirpath": str(ckpt_dir),
                        "filename": "sr-test",
                        "every_n_train_steps": 1,
                        "save_top_k": -1,
                    },
                }
            ],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    _run_cli(["fit", "--config", str(config_path)])

    ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
    assert ckpt_files, "fit did not write a checkpoint"
    return config_path, ckpt_files[-1]


def test_ckpt_path_round_trips_a_real_checkpoint(tiny_rgb_image_dir: Path, tmp_path: Path):
    """Regression: `--ckpt_path` must reload a checkpoint this project actually writes.

    Before the fix, ``SRLightning.__init__`` overwrote ``self._hparams`` with a
    '/'-flattened dict (built for clean TensorBoard HParams columns).
    ``LightningCLI._parse_ckpt_path`` reads the checkpoint's ``hyper_parameters``
    verbatim and re-parses it as CLI options (``model.<key>``), so a key like
    ``training_config/example_input_shape/0`` became an unparseable option name
    and every ``--ckpt_path`` invocation (fit resume, validate, test, export)
    raised ``SystemExit`` with "Parsing of ckpt_path hyperparameters failed!".

    Reloads the freshly-written checkpoint via ``validate --ckpt_path`` — the
    same code path ``fit`` (resume), ``test``, and ``export`` all share.
    """
    config_path, ckpt_path = _build_srcnn_checkpoint(tiny_rgb_image_dir, tmp_path)

    # Confirms the actual root cause is fixed, not just that nothing raised below:
    # hyper_parameters keys must not use the '/' TensorBoard-flattening separator,
    # which jsonargparse can't parse back as a CLI option name.
    raw = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    saved_hparams = raw["hyper_parameters"]
    assert saved_hparams, "checkpoint has no hyper_parameters"
    assert not any("/" in k for k in saved_hparams), (
        f"hyper_parameters keys must not contain '/': {list(saved_hparams)}"
    )

    # The actual regression: before the fix, this raised SystemExit from
    # LightningCLI._parse_ckpt_path failing to parse the '/'-keyed hparams.
    _run_cli(["validate", "--config", str(config_path), "--ckpt_path", str(ckpt_path)])


def test_ckpt_path_reloads_a_legacy_flattened_checkpoint(tiny_rgb_image_dir: Path, tmp_path: Path):
    """Regression: `--ckpt_path` must also reload checkpoints saved *before* the fix.

    This project already has long training runs whose only checkpoints predate
    the fix — resuming or testing them must keep working, not just newly-written
    checkpoints. Fabricates that scenario by taking a real checkpoint this branch
    writes and rewriting its ``hyper_parameters`` back into the pre-fix
    '/'-flattened shape (what ``SRLightning.__init__`` used to save, including the
    incidental ``processor``/``criterion`` entries), then reloads it exactly as
    ``sisr test``/``fit --ckpt_path`` would.
    """
    from sisr.training.lightning_module import SRLightning

    config_path, ckpt_path = _build_srcnn_checkpoint(tiny_rgb_image_dir, tmp_path)

    raw = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    legacy_hparams = SRLightning._flatten_hparams(raw["hyper_parameters"])
    legacy_hparams["processor"] = "RGBProcessor"
    legacy_hparams["criterion"] = "MSELoss"
    raw["hyper_parameters"] = legacy_hparams
    torch.save(raw, ckpt_path)

    # Must not raise — this is exactly the legacy-checkpoint scenario --ckpt_path failed on.
    _run_cli(["validate", "--config", str(config_path), "--ckpt_path", str(ckpt_path)])


def test_ckpt_path_preserves_eval_config_subclass_identity(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """`--ckpt_path` reload must keep `eval_config`'s subclass, not just its values.

    A regression here would revert to instantiating the field's bare annotation
    type (base ``SREvalConfig``) from the checkpoint's stored dict instead of
    merging onto the class the config file already selected
    (``SRResNetEvalConfig``) — the two assertions below can fail independently:
    the reloaded object's *class*, a subclass-only *default* the checkpoint's
    stored dict never mentions at all, and a stored value that differs from the
    one ``--config`` resolves to.

    That third assertion is what makes the other two non-vacuous: ``--config``
    alone already selects ``SRResNetEvalConfig`` and already yields
    ``ssim_impl='daala'``, so both checkpoints below have their stored
    ``crop_border`` rewritten to 7 — a value nothing in the config produces (the
    class default is 4). Deleting the merge entirely leaves 4 behind and fails.

    ``_parse_ckpt_path`` merges the reconstructed ``training_config``/
    ``eval_config`` dict through the *subcommand's own* parser
    (``self._parser(subcommand)``) onto the already-resolved
    ``model.init_args.eval_config`` namespace, whose ``class_path`` is already
    ``SRResNetEvalConfig`` from ``--config``. That merge fills in the given keys
    as ``init_args`` overrides on the class already selected, so a subclass-only
    default the checkpoint's dict doesn't mention (``ssim_impl``, demonstrated
    here since it's the field this arc added) still applies. Routing the same
    merge through the top-level, subcommand-dispatching parser instead forces
    jsonargparse to re-validate the whole subclass spec from scratch and fall
    back to the field's bare annotation type — the base class, wrong default.
    """
    # Pre-fix RED is a construction-time SystemExit ("model, processor"
    # required), raised before either assertion below runs — structural, not
    # untried: `model` is Union-typed and required with no fallback, so any
    # un-nested override hits it before ever reaching eval_config's resolution.
    from sisr.models.srresnet import SRResNetEvalConfig

    config_path, ckpt_path = _build_srresnet_checkpoint(tiny_rgb_image_dir, tmp_path)

    # Simulates a checkpoint saved before `ssim_impl` existed: the field simply
    # isn't a key in its stored eval_config dict (dataclasses.asdict at save
    # time wouldn't have produced one yet) — SRResNetEvalConfig's own
    # ssim_impl='daala' default must still apply on reload, not the base's
    # 'wang'.
    raw = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    current_hparams = {k: dict(v) for k, v in raw["hyper_parameters"].items()}
    current_hparams["eval_config"]["crop_border"] = 7
    legacy_hparams = {k: dict(v) for k, v in current_hparams.items()}
    del legacy_hparams["eval_config"]["ssim_impl"]

    legacy_ckpt_path = tmp_path / "legacy.ckpt"
    torch.save({**raw, "hyper_parameters": legacy_hparams}, legacy_ckpt_path)
    current_ckpt_path = tmp_path / "current.ckpt"
    torch.save({**raw, "hyper_parameters": current_hparams}, current_ckpt_path)

    cli = _run_cli(["validate", "--config", str(config_path), "--ckpt_path", str(legacy_ckpt_path)])
    assert isinstance(cli.model.eval_config, SRResNetEvalConfig)
    assert cli.model.eval_config.ssim_impl == "daala"
    assert cli.model.eval_config.crop_border == 7

    # A checkpoint saved by the current code carries `ssim_impl` explicitly
    # (dataclasses.asdict includes every current field) — confirms identity
    # holds for the ordinary round trip too, not just the missing-key case above.
    cli = _run_cli(
        ["validate", "--config", str(config_path), "--ckpt_path", str(current_ckpt_path)]
    )
    assert isinstance(cli.model.eval_config, SRResNetEvalConfig)
    assert cli.model.eval_config.ssim_impl == "daala"
    assert cli.model.eval_config.crop_border == 7


def test_reconstruct_ckpt_hparams_unflattens_legacy_keys():
    """Legacy '/'-flattened hyper_parameters reconstruct into nested
    training_config/eval_config dicts, lists restored from their '0'/'1'/...
    index encoding; model/*, processor, criterion, and _instantiator (never part
    of the loadable contract) are dropped."""
    from sisr.cli import _reconstruct_ckpt_hparams

    legacy = {
        "training_config/example_input_shape/0": 3,
        "training_config/example_input_shape/1": 24,
        "training_config/example_input_shape/2": 24,
        "training_config/init_strategy": "default",
        "eval_config/crop_border": 4,
        "eval_config/psnr_channels/0": "RGB",
        "eval_config/psnr_channels/1": "YCbCr",
        "model/scale": 4,
        "model/kernel_sizes/0": 9,
        "processor": "RGBSignedOutputProcessor",
        "criterion": "MSELoss",
        "_instantiator": "lightning.pytorch.cli.instantiate_module",
    }
    assert _reconstruct_ckpt_hparams(legacy) == {
        "training_config": {"example_input_shape": [3, 24, 24], "init_strategy": "default"},
        "eval_config": {"crop_border": 4, "psnr_channels": ["RGB", "YCbCr"]},
    }


def test_reconstruct_ckpt_hparams_passes_through_current_nested_format():
    """The current (post-fix) nested hyper_parameters format round-trips unchanged."""
    from sisr.cli import _reconstruct_ckpt_hparams

    nested = {
        "training_config": {"layer_lrs": None, "init_strategy": "paper"},
        "eval_config": {"crop_border": 3, "psnr_channels": ["RGB", "Y"]},
    }
    assert _reconstruct_ckpt_hparams(nested) == nested


# Resuming into the same checkpoint dirpath the first fit call already populated
# is exactly what a real resume does in production and is harmless here — silence
# the resulting benign UserWarning rather than route the resumed run to a second
# dirpath, which would be less faithful to how --ckpt_path resume is actually used.
@pytest.mark.filterwarnings("ignore:Checkpoint directory .* exists and is not empty.:UserWarning")
def test_fit_resume_via_ckpt_path_is_not_just_validate_test(
    tiny_rgb_image_dir: Path, tmp_path: Path
):
    """Regression: ``fit --ckpt_path`` (resume) shares the same broken code path as
    ``validate``/``test``/``export`` — ``_parse_ckpt_path`` runs unconditionally for
    every subcommand — and this project runs 1M-step jobs that resume from
    checkpoints, so resume specifically must keep working, not just fresh eval runs.
    """
    config_path, ckpt_path = _build_srcnn_checkpoint(tiny_rgb_image_dir, tmp_path)

    # A fresh cache_dir for the resumed run: reusing the first run's LMDB cache_dir
    # from a second Trainer in the same process trips a Windows file-lock/rebuild
    # race in LMDBCache (its still-open write env blocks the new readonly open) —
    # unrelated to this regression, so sidestep it rather than fight it here.
    resume_config = yaml.safe_load(config_path.read_text())
    resume_config["data"]["train_dataset"]["init_args"]["cache_dir"] = str(
        tmp_path / ".lmdb_cache_resume"
    )
    resume_config_path = tmp_path / "config_resume.yaml"
    resume_config_path.write_text(yaml.safe_dump(resume_config))

    # Must not raise; --trainer.max_steps=2 forces the resumed run past the
    # checkpoint's already-reached step 1, proving training actually continues.
    _run_cli(
        [
            "fit",
            "--config",
            str(resume_config_path),
            "--ckpt_path",
            str(ckpt_path),
            "--trainer.max_steps=2",
        ]
    )


def test_ckpt_path_reload_survives_subclass_mode(tmp_path, monkeypatch):
    """--ckpt_path must still seed model config after the subclass-mode move.

    _parse_ckpt_path injects the checkpoint's hyper_parameters as CLI options.
    In subclass mode those options live one level deeper (model.init_args.*):
    un-nested, the injected dict lands on `model` itself, which is Union-typed
    and required under subclass mode, so jsonargparse can't resolve it and
    raises SystemExit outright rather than quietly falling back.

    ``--ckpt_path`` is only a recognized option under subcommand-based parsing
    (``run=True``, the default) — ``run=False`` has no ``ckpt_path`` argument
    at all, so this reload can't be observed through ``_resolve``'s pattern.
    ``Trainer.validate`` is stubbed to a no-op so the subcommand completes
    without needing real data: hparams are already merged into ``cli.model``
    by the time ``instantiate_classes`` finishes, before ``validate`` runs.
    """
    import functools

    from sisr.cli import SRLightningCLI
    from sisr.training import SRDataModule, SRLightning

    # jsonargparse derives --ckpt_path/--verbose from Trainer.validate's signature
    # (add_method_arguments), so the stub must preserve it — functools.wraps sets
    # __wrapped__, which inspect.signature follows by default.
    @functools.wraps(lightning.pytorch.Trainer.validate)
    def _noop_validate(*args, **kwargs):
        return None

    monkeypatch.setattr(lightning.pytorch.Trainer, "validate", _noop_validate)

    # Every stored value here is one the resolved config does NOT produce:
    # SUBCLASS_MODE_CONFIG sets example_input_shape [3, 24, 24] and selects
    # SRResNetEvalConfig, whose crop_border default is 4. Storing the config's
    # own values instead would make the assertions below pass with the whole
    # merge deleted.
    ckpt = tmp_path / "fake.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                "training_config": {"example_input_shape": [3, 32, 32]},
                "eval_config": {"crop_border": 7},
            }
        },
        ckpt,
    )
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        SUBCLASS_MODE_CONFIG.replace("IMG_DIR", str(tmp_path).replace("\\", "/")),
        encoding="utf-8",
    )
    # See _resolve's comment: LightningCLI warns when both args= and pytest's own
    # sys.argv are set, and Lightning's own "No seed found" fires since
    # SUBCLASS_MODE_CONFIG carries no seed_everything: — both need local
    # suppression under the strict global filterwarnings=error.
    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message="GPU available but not used.*")
            warnings.filterwarnings("ignore", message="No seed found.*")
            cli = SRLightningCLI(
                model_class=SRLightning,
                datamodule_class=SRDataModule,
                subclass_mode_model=True,
                auto_configure_optimizers=False,
                save_config_callback=None,
                args=["validate", "--config", str(cfg), "--ckpt_path", str(ckpt)],
            )
    finally:
        sys.argv = saved_argv
    assert cli.model.eval_config.crop_border == 7
    # A tuple, not the stored list: the merge lands on the dataclass field, whose
    # annotation coerces it — more evidence the value came through the config
    # machinery rather than being read off the checkpoint dict directly.
    assert cli.model.training_config.example_input_shape == (3, 32, 32)
