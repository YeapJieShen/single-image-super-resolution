"""``sisr.models``' docstring names every architecture subpackage that exists."""

from pathlib import Path

import sisr.models

PACKAGE = Path(sisr.models.__file__).parent


def test_every_architecture_subpackage_is_named_in_the_package_docstring():
    subpackages = sorted(p.name for p in PACKAGE.iterdir() if (p / "__init__.py").exists())
    doc = sisr.models.__doc__ or ""

    assert subpackages  # the glob found something
    assert [name for name in subpackages if f"``{name}``" not in doc] == []
