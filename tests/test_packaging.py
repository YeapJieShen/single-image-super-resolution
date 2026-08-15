"""Packaging-contract checks that don't need the app imported."""

import re
import tomllib
import zipfile
from pathlib import Path

import hatchling.build

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def test_pyyaml_declared_in_dev_extra():
    """tests/test_cli.py imports `yaml`; PyYAML must be an explicit dev
    dependency, not a transitive accident of jsonargparse/lightning that could
    vanish on a resolver bump."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert any(re.match(r"pyyaml(\[|[<>=!~ ]|$)", spec.strip(), re.IGNORECASE) for spec in dev), (
        f"PyYAML missing from the dev extra: {dev}"
    )


def test_wheel_ships_only_the_package():
    """tests/reference/daala_ssim.c is a vendored C oracle for the test suite.
    The wheel must contain the package and nothing else, so test fixtures and
    reference sources never ship to users."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["sisr"]


def _floor(deps: list[str], name: str) -> tuple[int, ...]:
    for spec in deps:
        match = re.match(rf"{name}\s*>=\s*([0-9.]+)", spec.strip(), re.IGNORECASE)
        if match:
            return tuple(int(part) for part in match.group(1).split("."))
    raise AssertionError(f"{name} not found with a >= floor in: {deps}")


def test_torch_floor_excludes_weights_only_bypass():
    """torch.load(weights_only=True) is this project's entire load-time safety
    contract (sisr/cli.py, sisr/export.py). GHSA-53q9-r3pm-6pq6 / CVE-2025-32434
    lets that check be bypassed for arbitrary code execution on torch <= 2.5.1,
    fixed in 2.6.0 — so the floor must exclude the vulnerable range, not just
    exist."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    assert _floor(deps, "torch") >= (2, 6), f"torch floor must be >= 2.6: {deps}"
    assert _floor(deps, "torchvision") >= (0, 21), f"torchvision floor must be >= 0.21: {deps}"


def test_perceptual_extra_declares_lpips():
    """LPIPS needs the `lpips` package; torchmetrics gates it on _LPIPS_AVAILABLE.

    DISTS deliberately stays out of any extra — it needs only torchvision,
    already a core dependency.
    """
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert any(dep.startswith("lpips") for dep in extras["perceptual"])


def test_wheel_ships_py_typed(tmp_path):
    """sisr/py.typed (PEP 561) makes the package's type annotations a checked
    promise to consumers of the wheel, not just an internal convention.

    Builds a real wheel via hatchling's PEP 517 build_wheel() hook rather than
    only reading pyproject.toml's config, so a future include/exclude rule
    that silently drops the marker is caught here instead of in a user's
    environment.
    """
    wheel_name = hatchling.build.build_wheel(str(tmp_path))
    with zipfile.ZipFile(tmp_path / wheel_name) as wheel:
        assert "sisr/py.typed" in wheel.namelist()
