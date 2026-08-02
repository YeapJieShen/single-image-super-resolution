# single-image-super-resolution

[![test](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/test.yml/badge.svg)](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/test.yml)
[![build](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/build.yml/badge.svg)](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/build.yml)
[![coverage](https://codecov.io/gh/YeapJieShen/single-image-super-resolution/branch/main/graph/badge.svg)](https://app.codecov.io/gh/YeapJieShen/single-image-super-resolution)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

Reproduce single-image super-resolution papers without rewriting a training loop each
time. Every architecture runs through one `LightningCLI` entrypoint, and an experiment is
one self-contained YAML file.

## Install

Requires Python 3.12+.

```bash
pip install .
sisr --help
```

On a CUDA machine, install the GPU wheels from PyTorch's index first (they aren't on
PyPI), then the project:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install .
```

## Quickstart

Bring your own HR images. Training and benchmark sets are downloaded separately and are
not distributed here. Copy a template out of [`templates/`](templates/) and point its
dataset paths at wherever yours live.

```bash
# check the config resolves before committing to a run
sisr fit --config templates/config.srresnet.template.yaml --print_config

# train
sisr fit --config templates/config.srresnet.template.yaml

# evaluate a checkpoint against the benchmark sets
sisr test --config templates/config.srresnet.template.yaml --ckpt_path best.ckpt
```

Anything in the YAML can be overridden on the command line:

```bash
sisr fit --config my.yaml --trainer.max_steps=500000 --optimizer.init_args.lr=1e-3
```

PSNR/SSIM and bicubic│SR│HR image strips go to TensorBoard for every benchmark set, so you
can watch quality improve during a run.

## How it works

Three pieces compose, and none of them know about each other:

- **`SRModel`** is the network, a pure tensor function and nothing else.
- **`SRProcessor`** adapts colorspace and range between the dataset's RGB and whatever the
  model wants (Y channel, YCbCr, `[-1, 1]` RGB).
- **`SRTrainingConfig` / `SREvalConfig`** hold the per-paper knobs: learning-rate schedule,
  weight init, crop border, which colorspaces get scored.

`SRLightning` glues them together and `SRDataModule` feeds them. Adding an architecture
means writing a network and a config dataclass, not a new Lightning module. See
[CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-architecture).

## Models

| Model | Paper | LR input | Upsampling |
|---|---|---|---|
| SRCNN | [Dong et al. 2015](https://arxiv.org/pdf/1501.00092) | pre-upsampled to HR size | none, same-resolution refinement |
| SRResNet | [Ledig et al. 2017](https://arxiv.org/pdf/1609.04802) | true low-resolution | ×scale sub-pixel convolution |

## Comparability

Benchmark numbers only mean something if the inputs and the metrics match the papers', so
both are pinned:

- **LR generation** uses a vendored MATLAB-compatible `imresize`
  ([`sisr/imresize.py`](sisr/imresize.py), MIT, attribution in the file header) rather than
  OpenCV's bicubic. The two differ in ways that move PSNR: MATLAB antialiases by widening
  the kernel on downscale, and uses `a = -0.5` where OpenCV uses `a = -0.75`. Downscaling
  is verified byte-identical against the MATLAB-generated reference pairs distributed by
  the [EDSR authors](https://github.com/sanghyun-son/EDSR-PyTorch).
- **Metrics** are Y-channel PSNR/SSIM in BT.601 studio range (MATLAB's `rgb2ycbcr`
  convention, which is what published figures use), computed on output clamped to `[0, 1]`.

## ONNX export

```bash
pip install ".[export]"
sisr export --config my.yaml --ckpt_path best.ckpt --output_path model.onnx
```

The exported graph is the **bare model**, with no processor attached, and it accepts
arbitrary input sizes. For RGB models that graph is the whole pipeline. For Y-channel
models like SRCNN the consumer has to extract Y and recombine chroma itself; see
[`sisr/processors/y_channel.py`](sisr/processors/y_channel.py) for the reference
implementation. `sisr.export.to_onnx()` does the same thing from Python.

## Development

```bash
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE). All dependencies are MIT-compatible.
