"""The MLflow extra is optional: these run where it is NOT installed (CI), so no importorskip."""

import subprocess
import sys

import pytest


def test_importing_the_training_package_does_not_import_mlflow():
    code = "import sys, sisr.training; assert 'mlflow' not in sys.modules, 'mlflow imported'"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_a_missing_extra_is_an_actionable_error_not_a_bare_module_not_found(monkeypatch):
    from sisr.training import sinks

    monkeypatch.setitem(sys.modules, "mlflow", None)  # `import mlflow` now raises ImportError
    with pytest.raises(ImportError, match=r"pip install '\.\[mlflow\]'") as err:
        sinks._mlflow()
    assert not isinstance(err.value, ModuleNotFoundError)
