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

## Losses

The criterion is wired from YAML like everything else, and defaults to
`torch.nn.MSELoss`. Any `nn.Module` taking `(pred, target)` works, so L1 needs no
class of ours:

```yaml
model:
  criterion:
    class_path: torch.nn.L1Loss
```

| Class | What it is |
|---|---|
| `torch.nn.MSELoss` | the default; both papers' baseline |
| `torch.nn.L1Loss` | plain L1 |
| `sisr.losses.CharbonnierLoss` | `sqrt(diff² + eps²)` — L1 with a finite gradient at zero |
| `sisr.losses.TotalVariationLoss` | isotropic TV regulariser; ignores its target |
| `sisr.losses.VGG19FeatureLoss` | VGG19 feature-space perceptual loss, any `φ_{i,j}` — blocks 1-2 have 2 convolutions, blocks 3-5 have 4 (up to `vgg54`) |
| `sisr.losses.VGG16FeatureLoss` | the same on VGG16, but blocks 3-5 have 3 convolutions instead of 4 (deepest is `vgg53`) — experimentation only, no published SR recipe uses this depth |
| `sisr.losses.WeightedSumLoss` | named weighted sum of any of the above |

Combining them logs each term separately as `loss/train/{name}` and
`loss/val/{name}`, so you can see which one dominates:

```yaml
model:
  criterion:
    class_path: sisr.losses.WeightedSumLoss
    init_args:
      terms:
        vgg22:
          class_path: sisr.losses.VGG19FeatureLoss
          init_args:
            layer: vgg22
        tv:
          class_path: sisr.losses.TotalVariationLoss
      weights: {vgg22: 1.0, tv: 2.0e-8}
```

That is Ledig et al.'s SRResNet-VGG22 recipe: a VGG22 content loss plus total
variation at `2e-8`, no pixel term, trained from scratch. Three things to know
before running it:

- **The first use downloads ~548 MB** of VGG19 weights into the torch hub cache.
  Pass `weights: null` to skip the download, but only for tests — it builds a
  random VGG and warns, because the loss it then computes is meaningless.
- **TV's `2e-8` presumes a `[-1, 1]` model output range**, i.e.
  `RGBSignedOutputProcessor`. Under a `[0, 1]` processor the same image has half
  the total variation, so the effective weight differs by 2×.
- **A perceptual run scores worse PSNR by design.** The shipped templates already
  checkpoint on both `psnr/val/RGB` and `ssim/val/RGB` — under a perceptual
  criterion the PSNR-monitored "best" no longer tracks the training objective, so
  decide which monitored checkpoint you actually want.

A VGG loss normalises with RGB ImageNet statistics, so it refuses a 1-channel
processor (SRCNN's `YChannelProcessor`) unless you set `grayscale_to_rgb: true`,
and it refuses a 3-channel non-RGB processor (`YCbCrProcessor`) unless you set
`allow_non_rgb: true` — feeding Y/Cb/Cr planes to VGG as though they were R/G/B
trains on features that mean nothing. The frozen VGG is deliberately excluded
from checkpoints, so a `.ckpt` trained under one criterion loads into a module
configured with another.

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
- **SSIM has two incompatible conventions, and unlike PSNR the choice is not
  cosmetic.** PSNR is a closed-form function of squared error, so any correct
  implementation agrees with any other. SSIM depends on a local-window gaussian that
  the SR field never standardised on: Wang et al.'s original uses a fixed 11×11 window
  at sigma 1.5 — what `torchmetrics`, MATLAB's reference code, and BasicSR's
  `calculate_ssim` all compute, and therefore what most SR papers report. Ledig et al.
  (SRResNet/SRGAN) instead scored with the **daala** video-codec package, whose
  gaussian sigma scales with image height (`_h*(1.5/256)`) rather than staying fixed.
  The same image therefore scores differently under the two conventions, and a
  benchmark set's aggregate partly reflects the pixel dimensions of its images, not
  only reconstruction quality. [`sisr/ssim.py`](sisr/ssim.py) ports daala's method,
  verified against daala's own compiled C reference on 133 cases, and
  `SREvalConfig.ssim_impl` (`'wang'` or `'daala'`, see
  [`sisr/training/config.py`](sisr/training/config.py)) selects between them —
  `'wang'` is the base default, and
  [`SRResNetEvalConfig`](sisr/models/srresnet/config.py) overrides it to `'daala'`
  because that is the convention its paper used; SRCNN keeps `'wang'`, the field
  standard. The switch is **in place**: `ssim/val/RGB` and `ssim/val/Y` name the
  metric identically either way, and checkpoint filenames
  (`sr-{step}-ssim_val_RGB={value:.4f}.ckpt`, built by
  [`SRCheckpoint`](sisr/training/callbacks.py)) carry only the bare number — so
  neither the tag nor the filename reveals which convention produced a given value.
  It is recorded in `hparams` and in every artifact's `sisr_meta` instead. Consequently, an
  SRResNet SSIM figure is comparable to Ledig et al. and **not** to Wang-based tables
  (the EDSR/RCAN/SwinIR/BasicSR lineage); always say which convention a number came
  from.

## `--ckpt_path` silently drops subclass-only `eval_config` defaults

Checkpoints saved before `ssim_impl` existed do not pick up its new default when
reloaded — and the mechanism is not "the checkpoint overrides the CLI"; it is more
structural than that. `SRLightning` saves `eval_config` in `hparams` as
`dataclasses.asdict(self.eval_config)` — a bare dict with no `class_path`
([`sisr/training/lightning_module.py`](sisr/training/lightning_module.py)).
`SRLightningCLI._parse_ckpt_path` rebuilds that dict via `_reconstruct_ckpt_hparams`
and hands it to jsonargparse as the `model.eval_config` value
([`sisr/cli.py`](sisr/cli.py)). Because the dict carries no class identity,
jsonargparse instantiates the field's **annotation type** — the base `SREvalConfig`
— never the subclass (`SRResNetEvalConfig`) that actually produced the value, and it
does this whether or not a CLI override was also given, since the replacement runs
after argument parsing. Keys the dict happens to carry survive as explicit values;
any key it is missing falls back to `SREvalConfig`'s own default, not the subclass's.
A checkpoint saved before `ssim_impl` existed simply has no such key, so nothing
"overrides" anything — the field falls to the base default, `wang`. `crop_border=4`
and `psnr_channels=['RGB', 'Y']` survive reload only because `dataclasses.asdict`
happened to record them explicitly at save time, not because the subclass identity
is preserved.

**This generalises past `ssim_impl`.** Any `SREvalConfig` field an architecture
subclass overrides is equally at risk the moment a checkpoint predates that field, on
**every** subcommand that accepts `--ckpt_path` — `fit` resume included, not just
`validate`/`test`. That makes resume the more damaging case: resuming a pre-change
SRResNet run from an old checkpoint trains and checkpoints on `wang` `ssim/val/RGB`
for the rest of the run, while a fresh `fit` from the identical YAML uses `daala` —
two runs of "the same config" whose monitored metric means different things,
distinguishable only by reading `hparams`, not by anything in the logs or filenames.
This behaviour is locked in (deliberately not fixed) by
`test_ckpt_path_loses_subclass_only_eval_defaults` in
[`tests/test_cli.py`](tests/test_cli.py) — read it for the exact mechanics, including
why a CLI override alongside `--ckpt_path` doesn't help.

For example:

```bash
sisr validate --config my.yaml --ckpt_path old.ckpt --model.eval_config.ssim_impl=daala
```

silently scores with `wang`: the emitted config confirms `wang`, and the SSIM values
come back bit-identical to a pre-upgrade run. This is a one-time migration issue per
field, not an ongoing bug — a checkpoint written after a given field was added
carries it explicitly and restores it correctly; it is only checkpoints predating
that field that fall back.

Workaround: patch a **copy** of the checkpoint and re-score (or resume) from the
copy. Handle both hparams formats the codebase supports — current checkpoints nest
`eval_config` as a dict, but checkpoints from before `SRLightning` stopped
flattening `self.hparams` store `'/'`-joined keys instead (e.g.
`"eval_config/ssim_impl"`; see `_reconstruct_ckpt_hparams` in
[`sisr/cli.py`](sisr/cli.py) and the legacy-flattened-checkpoint test in
[`tests/test_cli.py`](tests/test_cli.py)) — and old checkpoints are exactly the
population this section targets:

```python
import torch

ckpt = torch.load("old.ckpt", weights_only=True, map_location="cpu")
hp = ckpt["hyper_parameters"]
if "eval_config" in hp and isinstance(hp["eval_config"], dict):
    hp["eval_config"]["ssim_impl"] = "daala"  # current nested format
else:
    hp["eval_config/ssim_impl"] = "daala"  # legacy '/'-flattened format
torch.save(ckpt, "old_daala.ckpt")
```

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
