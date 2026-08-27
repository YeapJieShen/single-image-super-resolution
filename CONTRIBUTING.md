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
mypy sisr                               # type-check, as CI runs it
```

`mypy` only checks the modules `[tool.mypy]`'s overrides don't blanket-ignore — Lightning's
hook system and jsonargparse's dynamic CLI/subclass resolution make `sisr/training/` and
`sisr/cli.py` resist useful static typing without a much larger, separate effort.

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

Branch off `main`, add tests with your change, run `pytest`, open a PR. Four checks have
to pass: **test** (pytest + coverage), **build** (`pip install .` and imports work),
**lint** (`ruff check` + `ruff format --check`), and **typecheck** (`mypy sisr`, over the
modules it doesn't blanket-ignore). `main` keeps linear history.

The path a change takes:

| Stage | What it means |
| --- | --- |
| **Design** | Anything touching a public contract, a metric, or the data path gets agreed before it gets written. A typo fix does not. |
| **Branch** | `<type>/<topic>`, e.g. `fix/cache-locking`, `docs/workflow`. |
| **Verify** | Correctness: a test that fails before your change and passes after. Performance: before/after numbers, on real data. |
| **PR** | Opened against `main`. Stacked PRs must be retargeted to `main` before their final push, or CI never runs on them. |
| **Merge** | Squash for a single-purpose PR; rebase for one carrying both code and docs, so the commits stay separate. |

### Evidence

Two claims need proof in the PR, not just assertion:

- **A bug fix** ships the test that catches it. Run it against the unfixed code and confirm
  it fails — a test that passes both ways documents nothing. Name it for the failure, not
  the fix.
- **A performance claim** ships before/after numbers measured on real data, reporting
  **medians with warm-up excluded**. A mean over a short run hides one-time setup costs
  (worker spawn, cache warm, autotune) and can invert the conclusion. Say how many
  iterations you discarded.

Anything touching `sisr/utils/imresize.py` must re-prove byte-identical output against the
MATLAB reference set; that byte-equality is the basis of every comparison to published
results.

### Commits

Conventional Commit types, with subjects that describe the effect:

```
<type>: <what changes, in the imperative>

<why — never a restatement of the diff>
```

- **Types:** `feat`, `fix`, `docs`, `perf`, `refactor`, `test`, `chore`.
- **Describe the effect, not the mechanism.**
  `fix: make the cache build lock unable to destroy live data`, not
  `fix: refactor _try_load`. Sentence case, no trailing period.
- **Code and docs may share a PR but never a commit.** Put docs in their own `docs:`
  commit. A PR carrying both is rebase-merged so the split survives.
- **Breaking changes** take `!` and a footer: `feat!: ...` plus
  `BREAKING CHANGE: <what downstream code must do>`.
- **The body explains why.** The diff already shows what. Omit it when the change is
  self-evident.
- Reference PRs (`#42`) freely; the history is public, so keep internal tracker ids out
  of it.

## Style

- Terse code. Comment the non-obvious *why*, never the *what*.
- Google-style docstrings on everything public in `sisr/`. Class docstrings document
  `__init__` args; `__init__` itself stays bare.
- Modern type hints: `list[str]`, `X | None`, and `collections.abc` for `Callable` /
  `Sequence`.
- No back-compat shims. Drop the old API, delete the dead branch.

## Adding a new architecture

You don't write a Lightning module. `SRLightning` already composes an `SRModel` with an
`SRProcessor`, fed by `SRDataModule`. The one exception is a second optimizer: adversarial
training drives two networks alternately, which Lightning's automatic loop cannot express,
so `SRGANLightning` subclasses `SRLightning` and owns its own training step. Anything that
trains one network needs no new module.

1. Subclass `sisr.models.base.SRModel`: `forward`, `self._hparams`, and optionally
   `reset_parameters(**kwargs)` for a paper-faithful init.
2. Add `<Arch>TrainingConfig` / `<Arch>EvalConfig` with the paper's defaults, copying the
   shape of `sisr/models/srcnn/config.py`.
3. Pick a processor (`RGBProcessor`, `RGBSignedOutputProcessor`, `YChannelProcessor`,
   `YCbCrProcessor`) or write one.
4. Copy a template, repoint the `class_path`s, run `sisr fit`.

## License

MIT. Dependencies are MIT-compatible. `sisr/utils/imresize.py` vendors an MIT-licensed
MATLAB-`imresize` port, so keep its attribution header intact.
