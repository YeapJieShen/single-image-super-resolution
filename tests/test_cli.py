"""End-to-end CLI smoke tests via subprocess."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "templates" / "config.srcnn.template.yaml"


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
    assert "class_path: sisr.models.srcnn.SRCNNTrainingConfig" in out
    assert "class_path: sisr.models.srcnn.SRCNNEvalConfig" in out
    assert "layer_lrs:" in out
    assert "crop_border: 3" in out
    assert "model_colorspace: Y" in out
    # Top-level optimizer block linked from YAML.
    assert "optimizer:" in out


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
