"""The static-typing gate: that it runs, that it bites, and what it covers."""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Modules whose dynamism (Lightning's hook system, jsonargparse's subclass
#: resolution) makes them fight a type checker rather than benefit from one.
#: Narrowing this list is the point; widening it needs a reason in the diff.
EXPECTED_IGNORED = {
    "sisr.training.callbacks",
    "sisr.training.gan_module",
    "sisr.training.lightning_module",
}


def _ignored_modules() -> set[str]:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return {
        module
        for override in config["tool"]["mypy"]["overrides"]
        if override.get("ignore_errors")
        for module in override["module"]
    }


def test_type_check_exemptions_are_named_one_by_one():
    """A glob like `sisr.training.*` silently swallows every module added to the
    package afterwards, including ones that would check clean. Listing them
    individually makes each exemption a visible line in a diff."""
    ignored = _ignored_modules()

    assert not [m for m in ignored if "*" in m], (
        f"exemptions must name modules, not globs: {sorted(ignored)}"
    )
    assert ignored == EXPECTED_IGNORED, (
        "the exemption list changed; if that is deliberate, update EXPECTED_IGNORED "
        "in this test in the same commit so it stays a decision rather than a drift"
    )


@pytest.mark.parametrize(
    "module",
    ["sisr.metrics.scoring", "sisr.training.config", "sisr.training.metadata"],
)
def test_modules_that_should_be_checked_are_not_exempt(module):
    """These sit next to exempt ones and would be easy to sweep back under a
    glob. `sisr.metrics.scoring` in particular was exempt purely because it used
    to live under `sisr/training/`."""
    assert module not in _ignored_modules()


@pytest.mark.skipif(shutil.which("mypy") is None, reason="mypy is not installed")
def test_the_gate_rejects_a_deliberate_violation(tmp_path):
    """A gate that has never been seen to fail is not known to be a gate.

    Runs the checker under this project's own config against a file that
    violates it, and requires the specific error rather than merely a non-zero
    exit -- which a crash, a bad path or an unreadable config would also give.
    """
    offender = tmp_path / "offender.py"
    offender.write_text(
        '"""Deliberately mistyped module."""\n\n\ndef f() -> int:\n    return "not an int"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(PYPROJECT), str(offender)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode != 0, f"the checker accepted a bad return type:\n{result.stdout}"
    assert "return-value" in result.stdout, (
        f"expected a return-value error, got:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.skipif(shutil.which("mypy") is None, reason="mypy is not installed")
def test_the_gate_accepts_the_same_module_once_corrected():
    """The counterpart to the test above: an error that fires on everything is
    not a gate either."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.py"
        good.write_text(
            '"""Correctly typed module."""\n\n\ndef f() -> int:\n    return 1\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--config-file", str(PYPROJECT), str(good)],
            capture_output=True,
            text=True,
            cwd=tmp,
        )

    assert result.returncode == 0, f"the checker rejected a clean module:\n{result.stdout}"
