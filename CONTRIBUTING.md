# Contributing

Thanks for your interest in `single-image-super-resolution` — a PyTorch Lightning
framework for reproducing single-image super-resolution (SISR) papers. This guide
covers how to set up a development environment, run the tests, and land a change.

## Development setup

Requires **Python ≥ 3.12**.

```bash
# Clone your fork, then install in editable mode with the dev extras:
git clone https://github.com/YeapJieShen/single-image-super-resolution.git
cd single-image-super-resolution
pip install -e ".[dev]"
```

The `[dev]` extra adds the test toolchain (`pytest`, `pytest-cov`). `pyproject.toml`
is the single source of truth for dependencies — there is no `requirements.txt`.

If you develop on a CUDA GPU, install the CUDA `torch` / `torchvision` wheels from
PyTorch's own index **first** (the `+cu###` wheels are not on PyPI), then install the
project — see the README's *GPU / CUDA notes*.

## Running the tests

```bash
# Full suite:
pytest

# With coverage (matches CI):
pytest --cov=sisr --cov-report=term
```

Tests live under `tests/`, mirroring the `sisr/` package layout
(e.g. `sisr/training/lightning_module.py` → `tests/training/test_lightning_module.py`).

### Strict-warnings policy

The suite runs under `filterwarnings = ["error", ...]`: **any unexpected warning fails
the test run.** A short, explicit ignore-list in `pyproject.toml` covers known upstream
deprecations only. If your change surfaces a new warning, fix the cause rather than
broadening the ignore-list; add a narrowly-scoped ignore only for a genuine third-party
deprecation you cannot control, and note why.

### Regression tests

When you fix a bug, add one test that **fails on the old behavior and passes on the new**.
Name it after what the failure catches, not after the fix.

## The change workflow

1. Branch off `main`.
2. Make your change; add or update tests alongside it.
3. Run `pytest` locally and make sure it is green.
4. Open a pull request against `main`.

All PRs must pass two GitHub Actions checks before they can merge:

- **test** — installs `.[dev]` and runs `pytest` with coverage (uploaded to Codecov).
- **build** — verifies `pip install .` and that `sisr` / `sisr.cli.main` import cleanly.

`main` uses linear history (rebase/squash, no merge commits) and requires PRs.

## Commit conventions

- **Imperative subject line** ("Add …", "Fix …", "Refactor …", "Document …").
- Where a commit closes a tracked item, reference it in the subject
  (e.g. a PR number like `(#12)`).
- **Keep documentation-only changes in commits separate from code changes.** A commit
  either touches code or touches docs, not both.

## Code style

- **Terse code, minimal comments.** Comment only non-obvious *why*, never narrate *what*.
- **Google-style docstrings** on every public class / function / method in `sisr/`
  (one-line summary opener, then `Args:` / `Returns:` / `Raises:` when the signature
  warrants). Class-level `Args:` documents `__init__`; `__init__` itself stays bare.
- **Modern type hints** — PEP 585 (`list`, `dict`, `tuple`) and PEP 604 (`X | None`);
  import `Callable` / `Sequence` from `collections.abc`.
- **No backward-compatibility shims when refactoring.** Prefer clean breaks — drop the
  old API, remove the unused parameter, delete the dead branch.

## Adding a new architecture

The training stack is generic: one `SRLightning` module composes an `SRModel`
(pure tensor network) with an `SRProcessor` (colorspace adapter), fed by a generic
`SRDataModule`. To add an architecture you do **not** write a new Lightning subclass —
instead:

1. Subclass `sisr.models.base.SRModel` with your network (`forward`, `self._hparams`,
   and an optional `reset_parameters(**kwargs)` init hook).
2. Define `<Arch>TrainingConfig` / `<Arch>EvalConfig` dataclasses with the paper's
   defaults (mirroring `sisr/models/srcnn/config.py`).
3. Pick an existing `SRProcessor` (`RGBProcessor`, `YChannelProcessor`, `YCbCrProcessor`)
   or add a new one.
4. Copy a template in `templates/`, point `model.model` / `model.processor` /
   `model.training_config` / `model.eval_config` and each dataset `class_path` at your
   new classes, and run `sisr fit --config <your.yaml>`.

## License

This project is MIT-licensed. Note it depends on
[AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX) (AGPL-3.0, or a
separate commercial license from upstream); if you redistribute or host this code as a
network service, review the AGPL-3.0 obligations.
