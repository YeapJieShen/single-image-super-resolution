# single-image-super-resolution

[![test](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/test.yml/badge.svg)](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/test.yml)
[![build](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/build.yml/badge.svg)](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/build.yml)
[![coverage](https://codecov.io/gh/YeapJieShen/single-image-super-resolution/branch/main/graph/badge.svg)](https://app.codecov.io/gh/YeapJieShen/single-image-super-resolution)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

`test`, `build`, `lint` and `typecheck` are required status checks: a pull request cannot merge
unless the suite passes on Linux and Windows (Python 3.12 and 3.13), the wheel and editable
installs work, `ruff` is clean and the package type-checks. A `py.typed` marker ships the
annotations.

Reproduce single-image super-resolution papers without rewriting a training loop each time. Every
architecture runs through one `LightningCLI` entrypoint (`sisr`), and an experiment is one
self-contained YAML file. LR images are derived with a byte-exact port of MATLAB's `imresize`, and
PSNR/SSIM follow each paper's own convention, so figures are comparable to the published ones.

**No pretrained checkpoint is released yet** ([licensing](docs/weights-and-licensing.md)).

## Install

Requires Python 3.12+.

```bash
pip install .                # on CUDA, first: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
sisr --help
```

Extras, each opt-in: `.[export]` (ONNX), `.[perceptual]` (LPIPS, NIQE, Ma, PI), `.[mlflow]`
(MLflow tracking and model registry), `.[dev]` (tests and lint).

## Quickstart

Datasets are not distributed here. Put HR images where the templates expect them:

```
data/
  DIV2K_train_HR/    DIV2K_valid_HR/                 # DIV2K, from its authors
  Set5_HR/  Set14_HR/  BSD100_HR/                    # benchmark sets
```

The benchmark archive and its pinned checksum are described under
[Comparability](docs/reproduction.md#comparability). Copy a template from
[`templates/`](templates/), then:

```bash
sisr fit --config templates/config.srresnet.template.yaml --print_config   # check it resolves
sisr fit --config templates/config.srresnet.template.yaml                  # train
sisr test --config templates/config.srresnet.template.yaml --ckpt_path best.ckpt
sisr fit --config my.yaml --trainer.max_steps=500000 --optimizer.init_args.lr=1e-3   # any key
```

A config that passes `--print_config` can still fail to instantiate, so smoke-run with
`--trainer.fast_dev_run=true` first. PSNR/SSIM and bicubic|SR|HR strips for every benchmark set go
to TensorBoard (and optionally MLflow). SRCNN on DIV2K-800 wants a 16 GB host
([why](docs/models.md#per-model)).

## How it works

`SRModel` (the network, a pure tensor function), `SRProcessor` (colorspace and range between RGB
and what the model wants) and `SRTrainingConfig`/`SREvalConfig` (per-paper knobs) compose without
knowing each other; `SRLightning` glues them and `SRDataModule` feeds them. A new architecture is a
network plus a config dataclass, never a new Lightning module
([CONTRIBUTING](CONTRIBUTING.md#adding-a-new-architecture)).

## Models

| Model | Paper | Parity with the paper |
|---|---|---|
| SRCNN | [Dong 2015](https://arxiv.org/pdf/1501.00092) | no publishable run |
| SRResNet | [Ledig 2017](https://arxiv.org/pdf/1609.04802) | reproduced, difference named |
| SRGAN | [Ledig 2017](https://arxiv.org/pdf/1609.04802) | not reproduced (cause isolated) |
| ESPCN | [Shi 2016](https://arxiv.org/pdf/1609.05158) | not yet reproduced |
| DRCN | [Kim 2016](https://arxiv.org/abs/1511.04491) | reproduced |
| VDSR | [Kim 2016](https://arxiv.org/abs/1511.04587) | reproduced |
| ESRGAN | [Wang 2018](https://arxiv.org/abs/1809.00219) | PSNR stage reproduced; GAN stage a recorded baseline |
| EDSR / EDSR+ | [Lim 2017](https://arxiv.org/abs/1707.02921) | reproduced |
| MDSR / MDSR+ | [Lim 2017](https://arxiv.org/abs/1707.02921) | within 0.16 dB, gaps named; not claimed as reproduced |

"Reproduced" means within 0.05 dB PSNR-Y of the paper on Set5, Set14 and BSD100, using the
authors' released weights where they exist; "difference named" means within 0.20 dB with the cause stated. [`docs/models.md`](docs/models.md) has each model's
input contract, template and parity detail.

## Read before you trust a number

A figure is comparable only to one computed the same way. `psnr/val/Y` is BT.601 **studio-range**
luma on output clamped to `[0, 1]`. SSIM has two conventions (`wang`, the field standard, and
`daala`, which Ledig et al. used; SRResNet and SRGAN default to `daala`), and several papers'
MATLAB scripts round to 8 bits, score single-channel images raw or degrade LR in float; each
template sets what its paper did. Details:
[`docs/reproduction.md#comparability`](docs/reproduction.md#comparability) and
[`docs/configuration.md`](docs/configuration.md#eval_config-and-the-two-ssim-conventions).

## Where to go next

- [`docs/configuration.md`](docs/configuration.md): every YAML key, SRGAN training, step axes
  (`global_step` versus batches), command-line overrides, MLflow, export.
- [`docs/models.md`](docs/models.md): the model table in full.
- [`docs/reproduction.md`](docs/reproduction.md): measurements, protocols and conventions.
- [`docs/weights-and-licensing.md`](docs/weights-and-licensing.md): weights and data policy.
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CHANGELOG.md`](CHANGELOG.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT, see [LICENSE](LICENSE). All dependencies are MIT-compatible.
