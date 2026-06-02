"""End-to-end CLI smoke tests via subprocess."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "templates" / "config.srcnn.template.yaml"
SRRESNET_TEMPLATE = REPO_ROOT / "templates" / "config.srresnet.template.yaml"


def _cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Invoke `python -m sisr.cli ...` from the repo root with a fresh process."""
    return subprocess.run(
        [sys.executable, "-m", "sisr.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_print_config_resolves():
    """`cli fit --print_config` exits 0 and resolves the SRCNN template."""
    proc = _cli("fit", "--config", str(TEMPLATE), "--print_config")
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    out = proc.stdout
    # Sanity markers — confirm config-class wiring resolved correctly.
    assert "class_path: sisr.models.srcnn.SRCNN" in out
    assert "class_path: sisr.processors.YChannelProcessor" in out
    assert "class_path: sisr.models.srcnn.SRCNNTrainingConfig" in out
    assert "class_path: sisr.models.srcnn.SRCNNEvalConfig" in out
    assert "layer_lrs:" in out
    assert "crop_border: 3" in out
    # The removed model_colorspace field must not reappear.
    assert "model_colorspace" not in out
    # Top-level optimizer block linked from YAML.
    assert "optimizer:" in out


def test_cli_srresnet_print_config_resolves():
    """`cli fit --print_config` exits 0 and resolves the SRResNet template
    (model + base configs + random-crop dataset classes)."""
    proc = _cli("fit", "--config", str(SRRESNET_TEMPLATE), "--print_config")
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    out = proc.stdout
    assert "class_path: sisr.models.srresnet.SRResNet" in out
    assert "class_path: sisr.processors.RGBProcessor" in out
    assert "class_path: sisr.datasets.srresnet.TrainDataset" in out
    assert "class_path: sisr.datasets.srresnet.ValidationDataset" in out
    assert "hr_crop_size: 96" in out
    assert "crop_border: 4" in out
    # The removed model_colorspace field must not reappear.
    assert "model_colorspace" not in out


def test_cli_test_subcommand_exposes_ckpt_path():
    """`cli test --help` documents --ckpt_path and --data.test_datasets."""
    proc = _cli("test", "--help")
    assert proc.returncode == 0
    assert "--ckpt_path" in proc.stdout
    assert "--data.test_datasets" in proc.stdout


def test_cli_optimizer_lr_override():
    """Top-level --optimizer.init_args.lr override surfaces in resolved config."""
    proc = _cli(
        "fit", "--config", str(TEMPLATE),
        "--optimizer.init_args.lr=5e-3",
        "--print_config",
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "lr: 0.005" in proc.stdout


def test_cli_help_lists_subcommands():
    proc = _cli("--help")
    assert proc.returncode == 0
    for sub in ("fit", "validate", "test", "predict"):
        assert sub in proc.stdout


def test_cli_matmul_precision_accepted_and_surfaces():
    """`--matmul_precision=high` is accepted and round-trips through --print_config."""
    proc = _cli(
        "fit", "--config", str(TEMPLATE),
        "--matmul_precision=high",
        "--print_config",
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "matmul_precision: high" in proc.stdout


def test_cli_matmul_precision_defaults_to_null():
    """When unset, matmul_precision surfaces as null in --print_config."""
    proc = _cli("fit", "--config", str(TEMPLATE), "--print_config")
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "matmul_precision: null" in proc.stdout


def test_cli_matmul_precision_rejects_invalid():
    """Invalid matmul_precision values are rejected by the Literal validator."""
    proc = _cli(
        "fit", "--config", str(TEMPLATE),
        "--matmul_precision=bogus",
        "--print_config",
    )
    assert proc.returncode != 0


# In-process unit tests for SRLightningCLI.before_instantiate_classes. Subprocess
# tests above cover argparse wiring; these cover the hook's branching logic so it
# shows up in line coverage.
def _make_cli_stub(subcommand: str | None, matmul_precision: str | None):
    """Build an SRLightningCLI instance bypassing __init__ for direct hook testing."""
    from types import SimpleNamespace
    from sisr.cli import SRLightningCLI

    cli = SRLightningCLI.__new__(SRLightningCLI)
    cli.subcommand = subcommand
    cli.config = {subcommand: SimpleNamespace(matmul_precision=matmul_precision)} if subcommand else {}
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

    The resolved --print_config output is the source of truth; checking the YAML
    file directly would miss cases where a default is inherited or overridden.
    """
    proc = _cli("fit", "--config", str(template_path), "--print_config")
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "default_hp_metric: false" in proc.stdout, (
        f"{template_path.name} does not disable the hp_metric placeholder. "
        f"Add `default_hp_metric: false` under the TensorBoardLogger init_args."
    )
