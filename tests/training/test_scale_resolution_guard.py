"""One implementation of "the run's scale": training_config.scale, else the model's hparam."""

import ast
from pathlib import Path

SISR = Path(__file__).resolve().parents[2] / "sisr"


def _reads_config_scale(fn: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Attribute)
        and n.attr == "scale"
        and isinstance(n.value, ast.Attribute)
        and n.value.attr == "training_config"
        for n in ast.walk(fn)
    )


def _reads_hparam_scale(fn: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "get"
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "hparams"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "scale"
        for n in ast.walk(fn)
    )


def test_the_scale_fallback_pair_lives_in_exactly_one_function():
    holders = []
    for path in sorted(SISR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and _reads_config_scale(node):
                if _reads_hparam_scale(node):
                    holders.append(f"{path.relative_to(SISR).as_posix()}:{node.name}")

    assert holders == ["training/lightning_module.py:resolved_scales"]
