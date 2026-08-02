# Contributing

## Setup

Requires Python 3.12+.

```bash
git clone https://github.com/YeapJieShen/single-image-super-resolution.git
cd single-image-super-resolution

pip install -e ".[dev]"
```

[uv](https://github.com/astral-sh/uv) is a faster alternative if you prefer it:
`python -m uv pip install -e ".[dev]"`.

On a CUDA GPU, install the `torch` / `torchvision` wheels from PyTorch's index first, as
described in the [README](README.md#install).

## Tests

```bash
pytest                                  # full suite
pytest --cov=sisr --cov-report=term     # with coverage, as CI runs it
```

`tests/` mirrors `sisr/`, so `sisr/training/lightning_module.py` maps to
`tests/training/test_lightning_module.py`.

Warnings are errors (`filterwarnings = ["error", ...]`). If your change raises a new one,
fix the cause. Only add an ignore for a third-party deprecation you can't control, with a
comment saying why.

When you fix a bug, add a test that fails before the fix and passes after. Name it after
the failure it catches.

**Debugging a dataset?** The templates use `num_workers: 16`, so `__getitem__` runs in
subprocesses where breakpoints never fire. Run with
`--data.train_dataloader_kwargs.num_workers=0` to get it back in-process.

## Making a change

Branch off `main`, add tests with your change, run `pytest`, open a PR. Three checks have
to pass: **test** (pytest + coverage), **build** (`pip install .` and imports work), and
**lint** (`ruff check` + `ruff format --check`). `main` keeps linear history.

Commits:

- Imperative subject, for example "Add ...", "Fix ...", "Remove ...".
- Docs-only changes go in their own commit, never mixed with code.

## Style

- Terse code. Comment the non-obvious *why*, never the *what*.
- Google-style docstrings on everything public in `sisr/`. Class docstrings document
  `__init__` args; `__init__` itself stays bare.
- Modern type hints: `list[str]`, `X | None`, and `collections.abc` for `Callable` /
  `Sequence`.
- No back-compat shims. Drop the old API, delete the dead branch.

## Adding a new architecture

You don't write a Lightning module. `SRLightning` already composes an `SRModel` with an
`SRProcessor`, fed by `SRDataModule`.

1. Subclass `sisr.models.base.SRModel`: `forward`, `self._hparams`, and optionally
   `reset_parameters(**kwargs)` for a paper-faithful init.
2. Add `<Arch>TrainingConfig` / `<Arch>EvalConfig` with the paper's defaults, copying the
   shape of `sisr/models/srcnn/config.py`.
3. Pick a processor (`RGBProcessor`, `RGBSignedOutputProcessor`, `YChannelProcessor`,
   `YCbCrProcessor`) or write one.
4. Copy a template, repoint the `class_path`s, run `sisr fit`.

## License

MIT. Dependencies are MIT-compatible. `sisr/imresize.py` vendors an MIT-licensed
MATLAB-`imresize` port, so keep its attribution header intact.
