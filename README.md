# single-image-super-resolution

[![coverage](https://codecov.io/gh/YeapJieShen/single-image-super-resolution/branch/main/graph/badge.svg)](https://app.codecov.io/gh/YeapJieShen/single-image-super-resolution)

**Reproduce single-image super-resolution (SISR) papers with one command.** Train any
architecture through a single `LightningCLI` entrypoint — switching models is just
pointing the config at a different network and dataset. Built on PyTorch Lightning; every
experiment is one self-contained YAML.

## Why this exists

SISR papers each ship their own training script, colorspace quirks, and evaluation
protocol. This framework factors all of that into reusable parts — a generic
`SRLightning` module (an `SRModel` network × an `SRProcessor` colorspace adapter), a
generic `SRDataModule`, and per-paper behavior expressed in config dataclasses — so
reproducing a paper means writing a config, not a new training loop.

## Models

| Model | Paper | LR input | Upsampling | Colorspace |
|---|---|---|---|---|
| **SRCNN** | [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092) | Pre-upsampled to HR size (blur → downsample → bicubic up) | None — same-resolution refinement | Y channel |
| **SRResNet** | [Photo-Realistic Single Image Super-Resolution Using a GAN](https://arxiv.org/pdf/1609.04802) | Genuine low-resolution | ×`scale` sub-pixel convolution | RGB |

During training and test evaluation, PSNR/SSIM are logged and **bicubic│SR│HR** image
strips are written to TensorBoard for each benchmark set (Set5, Set14, …), so you can
watch reconstruction quality improve run-over-run.

## Quickstart

Requires **Python ≥ 3.12**.

```bash
# 1. Install (CPU / default):
pip install .

# 2. Confirm the CLI is wired up:
sisr --help

# 3. Put HR images under data/ (e.g. data/DIV2K_train_HR, data/Set5_HR, …),
#    then validate a config resolves without training:
sisr fit --config templates/config.srcnn.template.yaml --print_config

# 4. Train:
sisr fit --config templates/config.srcnn.template.yaml
```

`--print_config` dumps the fully resolved config and exits — a fast way to check a config
before committing to a run (no dataset build happens at parse time).

## Usage

Training is configured entirely through YAML. Copy a template from
[`templates/`](templates/), edit the dataset directories and experiment output paths,
then:

```bash
# Train SRCNN (x3):
sisr fit --config templates/config.srcnn.template.yaml

# Train SRResNet (x4):
sisr fit --config templates/config.srresnet.template.yaml
```

Per-run overrides can be passed on the CLI without editing the YAML (dot-notation;
optimizer/scheduler args are top-level keys):

```bash
sisr fit --config templates/config.srcnn.template.yaml \
    --trainer.max_steps=500000 --optimizer.init_args.lr=1e-3
```

After training, evaluate a checkpoint against the test sets:

```bash
sisr test --config templates/config.srcnn.template.yaml \
    --ckpt_path path/to/best.ckpt
```

Test sets are also surfaced during `fit` (monitored each validation cycle). The `sisr`
console script is registered by `pyproject.toml`; `python -m sisr.cli ...` works too.

## GPU / CUDA notes

The CUDA-built `torch` / `torchvision` wheels aren't on PyPI, so GPU users install them
from PyTorch's own index **first** (pick the `cu###` matching your CUDA toolkit), then
install the project — the second command sees `torch` already satisfied and resolves the
rest from PyPI. Same convention as PyTorch's own
[Get Started](https://pytorch.org/get-started/locally/) selector.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install .

# or, with uv (inside the activated environment):
python -m uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m uv pip install .
```

The bundled templates set `accelerator: cuda`; change it to `cpu` to run without a GPU.

## Extending it

The training stack is architecture-agnostic — adding a model does **not** require a new
Lightning subclass:

1. Subclass `sisr.models.base.SRModel` with your network.
2. Define `<Arch>TrainingConfig` / `<Arch>EvalConfig` dataclasses with the paper's
   defaults (see `sisr/models/srcnn/config.py`).
3. Pick an `SRProcessor` (`RGBProcessor`, `YChannelProcessor`, `YCbCrProcessor`) or add
   one.
4. Copy a template, point the `class_path`s at your new classes, and `sisr fit`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

## Development

```bash
pip install -e ".[dev]"     # or: python -m uv pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev loop, testing conventions, commit
style, and code-style expectations.

## Dependency notes

`pyproject.toml` is the single source of truth for dependencies (no `requirements.txt`).

This project depends on
[AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX), licensed under
AGPL-3.0 (or a separate commercial license from upstream). The
`single-image-super-resolution` project itself remains MIT-licensed. Users who
redistribute or host this code as a network service should review AGPL-3.0 obligations.
AlbumentationsX is capped at `<=2.1.0` so it stays on the `simsimd`-backed `albucore`;
`albucore` 0.1.0 switched its native backend to one whose runtime dependency
(`libomp140`) is not bundled on stock Windows / Python 3.13, breaking import there.

## License

MIT — see [LICENSE](LICENSE).
