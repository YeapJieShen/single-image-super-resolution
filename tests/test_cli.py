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

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "templates" / "config.srcnn.template.yaml"
SRRESNET_TEMPLATE = REPO_ROOT / "templates" / "config.srresnet.template.yaml"


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

    # LightningCLI warns when both `args=` and sys.argv[1:] are set (it sees
    # pytest's own argv). The strict `filterwarnings=["error"]` would turn that
    # into an error, so blank sys.argv for the duration of the parse.
    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        # Two framework-internal warnings fire only on the in-process path (the
        # subprocess --print_config path never surfaced them) and would trip the
        # global strict "error" filter, so suppress them locally — leaving the
        # strict filter intact for our own code:
        #   * jsonargparse DeprecationWarning: lightning's _add_instantiators
        #     calls add_instantiator / instantiate_classes, deprecated in 4.49.
        #   * "GPU available but not used": run=False instantiates the Trainer and
        #     we force accelerator=cpu; harmless on a GPU dev box, absent on CI.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message="GPU available but not used.*")
            return SRLightningCLI(
                model_class=SRLightning,
                datamodule_class=SRDataModule,
                auto_configure_optimizers=False,
                save_config_kwargs={"overwrite": True},
                args=[*args, "--trainer.accelerator=cpu", "--trainer.devices=1"],
                run=False,
            )
    finally:
        sys.argv = saved_argv


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
    from sisr.processors import RGBProcessor

    cli = _resolve("--config", str(SRRESNET_TEMPLATE))
    m = cli.model
    assert isinstance(m.model, SRResNet)
    assert isinstance(m.processor, RGBProcessor)
    assert isinstance(m.training_config, SRResNetTrainingConfig)
    assert isinstance(m.eval_config, SRResNetEvalConfig)
    assert m.eval_config.crop_border == 4  # inherited-default check
    # Dataset specs stay plain {class_path, init_args} dicts (materialized lazily
    # in SRDataModule.setup), so assert on the resolved raw config.
    assert cli.config.data.train_dataset["class_path"] == "sisr.datasets.srresnet.TrainDataset"
    assert cli.config.data.val_dataset["class_path"] == "sisr.datasets.srresnet.ValidationDataset"
    assert cli.config.data.train_dataset["init_args"]["hr_crop_size"] == 96
    # The removed model_colorspace field must not reappear anywhere.
    assert not hasattr(m, "model_colorspace")
    assert not hasattr(m.eval_config, "model_colorspace")


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
            auto_configure_optimizers=False,
            save_config_kwargs={"overwrite": True},
            args=["test", "--help"],
            run=True,
        )
    out = capsys.readouterr().out
    assert "--ckpt_path" in out
    assert "--data.test_datasets" in out


def test_optimizer_lr_override_in_process():
    """Top-level --optimizer.init_args.lr override surfaces in the resolved config."""
    cli = _resolve("--config", str(TEMPLATE), "--optimizer.init_args.lr=5e-3")
    assert cli.config.optimizer.init_args.lr == pytest.approx(0.005)


def test_matmul_precision_accepted_in_process():
    """--matmul_precision=high is accepted and round-trips into the resolved config."""
    cli = _resolve("--config", str(TEMPLATE), "--matmul_precision=high")
    assert cli.config.matmul_precision == "high"


def test_matmul_precision_defaults_to_none_in_process():
    """When unset, matmul_precision resolves to None."""
    cli = _resolve("--config", str(TEMPLATE))
    assert cli.config.matmul_precision is None


def test_matmul_precision_rejects_invalid_in_process():
    """Invalid matmul_precision is rejected by the Literal validator during parse
    (SystemExit from jsonargparse), before any class is instantiated."""
    with pytest.raises(SystemExit):
        _resolve("--config", str(TEMPLATE), "--matmul_precision=bogus")


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


@pytest.mark.parametrize("template_path", TEMPLATE_PATHS, ids=[p.name for p in TEMPLATE_PATHS])
def test_template_disables_default_hp_metric(template_path: Path):
    """Each shipped template must disable TensorBoard's hp_metric: -1 placeholder.

    Resolved in-process: cli.config is the same merged config --print_config would
    dump, so an inherited/overridden default is honored (checking the raw YAML
    would miss that)."""
    cli = _resolve("--config", str(template_path))
    loggers = cli.config.trainer.logger
    tb = next(logger for logger in loggers if str(logger.class_path).endswith("TensorBoardLogger"))
    assert tb.init_args.default_hp_metric is False, (
        f"{template_path.name} does not disable the hp_metric placeholder. "
        f"Add `default_hp_metric: false` under the TensorBoardLogger init_args."
    )
