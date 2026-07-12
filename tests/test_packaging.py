"""Packaging-contract checks that don't need the app imported."""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyyaml_declared_in_dev_extra():
    """tests/test_cli.py imports `yaml`; PyYAML must be an explicit dev
    dependency, not a transitive accident of jsonargparse/lightning that could
    vanish on a resolver bump."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert any(re.match(r"pyyaml(\[|[<>=!~ ]|$)", spec.strip(), re.IGNORECASE) for spec in dev), (
        f"PyYAML missing from the dev extra: {dev}"
    )
