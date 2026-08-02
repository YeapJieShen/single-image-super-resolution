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
| **SRCNN** | [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092) | Pre-upsampled to HR size (bicubic down → bicubic up) | None — same-resolution refinement | Y channel |
| **SRResNet** | [Photo-Realistic Single Image Super-Resolution Using a GAN](https://arxiv.org/pdf/1609.04802) | Genuine low-resolution | ×`scale` sub-pixel convolution | RGB |

During training and test evaluation, PSNR/SSIM are logged and **bicubic│SR│HR** image
strips are written to TensorBoard for each benchmark set (Set5, Set14, …), so you can
watch reconstruction quality improve run-over-run.

### Degradation protocol (how LR is generated)

Both datasets derive their LR input from HR via bicubic resizing, selectable per-dataset
via `resize_backend`:

- **`'matlab'` (default, tracked templates use it).** A vendored (not a dependency),
  MATLAB-`imresize`-compatible bicubic resize — antialiased (MATLAB widens the
  interpolation kernel by the scale factor on downscale; this widening *is* the
  low-pass, so no separate blur step is used or accepted with this backend) and using
  MATLAB's own bicubic coefficient (a=-0.5, vs. OpenCV's a=-0.75). This makes the LR
  **inputs** directly comparable to published papers, most of which generate their
  benchmark LR images with real MATLAB. See `sisr/imresize.py` for the implementation
  and its license/attribution header, and `tests/test_imresize.py` for verification.
  Byte-exact inputs alone don't make **reported** PSNR/SSIM comparable, though:
  Y-channel figures here use BT.601 full-range Y (`sisr/colorspace.py`), while the
  literature's published Y-channel numbers use MATLAB `rgb2ycbcr`'s studio-range
  convention — a known, exact `20·log10(255/219) ≈ 1.3225 dB` offset, not an unknown one.
- **`'cv2'` (opt-in).** Plain `cv2.INTER_CUBIC`, no antialiasing of its own. SRCNN's
  `blur_sigma` (a pre-resize Gaussian blur) only has an effect on this path — passing
  `blur_sigma` together with `resize_backend='matlab'` raises `ValueError` at
  construction, since MATLAB's kernel widening already *is* the low-pass and stacking
  an explicit blur on top would push PSNR away from published values, not toward them.
  This backend exists solely to keep LMDB caches built before the `'matlab'` backend
  existed reproducible; new work should use the default.

**Honest limit:** without access to real MATLAB, this project cannot prove
byte-identity with MATLAB's `imresize` from first principles. `tests/test_imresize.py`
asserts byte-exact reproduction against the standard Set5/Set14 bicubic LR pairs
distributed by the EDSR authors (themselves generated with real MATLAB) — the
strongest available claim short of running MATLAB itself. That test is skipped
(not a failure) when the reference archive hasn't been fetched locally; see the
test file's module docstring for the source URL and checksum.

Switching `resize_backend` (or the one-time move from AlbumentationsX to
`sisr.imresize`) invalidates previously-built LMDB caches and renumbers any
previously recorded benchmark figure — expected, one-time costs of the change,
not a bug.

### Reproducibility note

The tracked templates set `seed_everything: 42` **and** `trainer.deterministic: false`.
That pairing is deliberate: `deterministic: false` allows cuDNN to pick
non-deterministic (but faster) convolution algorithms, so **runs are not
bit-for-bit reproducible** even with the same seed — a throughput trade worth
knowing about explicitly in a paper-*reproduction* project, where "reproduction"
means comparable metrics, not bit-identical checkpoints.

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

## ONNX export

Trained models can be exported to ONNX for inference outside this framework — install
the optional extra first:

```bash
pip install ".[export]"     # or: python -m uv pip install ".[export]"

sisr export --config templates/config.srresnet.template.yaml \
    --ckpt_path path/to/best.ckpt --output_path model.onnx
```

`sisr.export.to_onnx(...)` is the underlying function, for exporting from Python
directly (e.g. right after `trainer.fit(...)`, without a checkpoint round-trip):

```python
from sisr.export import to_onnx

to_onnx(sr_lightning_module, "model.onnx")
```

**What gets exported — the bare model, not the processor.** The graph is exactly
`SRLightning.forward`: the wrapped `SRModel`, with no `SRProcessor` colorspace step.
Consumers are expected to have `sisr` importable and call `processor.extract` /
`processor.reconstruct` themselves — that keeps the graph honest about what it actually
computes, rather than silently baking in Python-side pre/post-processing an ONNX runtime
can't see.

- **SRResNet with `RGBProcessor` is unaffected by this**: `extract` / `reconstruct` are
  identity functions, so the exported graph already is the complete LR-RGB → SR-RGB
  pipeline.
- **SRResNet with `RGBSignedOutputProcessor` needs one step**: that processor trains the
  model to emit `[-1, 1]` (the SRGAN paper's HR range), so the graph's output must be
  rescaled by `(out + 1) / 2` to land back in `[0, 1]`. The input side needs nothing —
  the model consumes `[0, 1]` either way.
- **SRCNN is not**: it trains on the Y channel, so the exported graph only maps Y → Y. A
  non-Python consumer of an SRCNN ONNX graph must reimplement the surrounding steps
  itself — extract Y from the LR RGB image (`sisr.colorspace.rgb_to_ycbcr`), run the
  graph, then bicubic-upsample the LR Cb/Cr channels back to the SR size and recombine
  (`sisr.colorspace.ycbcr_to_rgb`). See `sisr/processors/y_channel.py` for the exact
  reference implementation.

The exported graph accepts **arbitrary spatial dimensions** (`dynamic_axes` on height and
width) — it is not limited to the size in `training_config.example_input_shape`, which is
only a TensorBoard-graph-logging dummy input.

**Deploying to TensorRT.** This project does not depend on TensorRT (NVIDIA-only,
version-brittle, and untestable in CI without a GPU runner). Convert the exported `.onnx`
yourself with NVIDIA's `trtexec`, which ships with the TensorRT SDK:

```bash
trtexec --onnx=model.onnx --saveEngine=model.trt \
    --minShapes=input:1x3x64x64 --optShapes=input:1x3x256x256 --maxShapes=input:1x3x1080x1920
```

(drop the batch/channel dims to match SRCNN's single-channel Y input). `--minShapes` /
`--maxShapes` are required because of the dynamic H/W axes above.

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
3. Pick an `SRProcessor` (`RGBProcessor`, `RGBSignedOutputProcessor`,
   `YChannelProcessor`, `YCbCrProcessor`) or add one.
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
All dependencies are MIT-compatible; the project itself is MIT-licensed (see below).
`sisr/imresize.py` vendors (does not depend on) a small MIT-licensed MATLAB-`imresize`
port — see that file's header for the source and attribution.

## License

MIT — see [LICENSE](LICENSE).
