# single-image-super-resolution

[![test](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/test.yml/badge.svg)](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/test.yml)
[![build](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/build.yml/badge.svg)](https://github.com/YeapJieShen/single-image-super-resolution/actions/workflows/build.yml)
[![coverage](https://codecov.io/gh/YeapJieShen/single-image-super-resolution/branch/main/graph/badge.svg)](https://app.codecov.io/gh/YeapJieShen/single-image-super-resolution)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

Those badges aren't decorative. `test`, `build`, `lint` and `typecheck` are all
**required status checks**, so a pull request cannot merge unless the full suite
passes on Linux and Windows across Python 3.12 and 3.13, the wheel and the editable
install both work, `ruff` is clean, and the package type-checks. The package also
ships a `py.typed` marker, so those annotations reach you rather than stopping at
the repository boundary.

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
dataset paths at wherever yours live. Every setting a template exposes is explained in
[`docs/configuration.md`](docs/configuration.md).

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
| SRGAN | [Ledig et al. 2017](https://arxiv.org/pdf/1609.04802) | true low-resolution | ×scale sub-pixel convolution |

SRGAN's network *is* SRResNet — only how it is trained differs. See [SRGAN](#srgan).

## Losses

The criterion is wired from YAML like everything else, and defaults to
`torch.nn.MSELoss`. Any `nn.Module` taking `(pred, target)` works, so L1 needs no
class of ours:

```yaml
model:
  init_args:
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
  init_args:
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
- **A perceptual run scores worse PSNR by design.** The shipped non-adversarial
  templates (SRCNN, SRResNet) already checkpoint on both `psnr/val/RGB` and
  `ssim/val/RGB` — under a perceptual criterion the PSNR-monitored "best" no longer
  tracks the training objective, so decide which monitored checkpoint you actually
  want. (SRGAN's template is deliberately monitor-free instead — see [SRGAN](#srgan).)

A VGG loss normalises with RGB ImageNet statistics, so it refuses a 1-channel
processor (SRCNN's `YChannelProcessor`) unless you set `grayscale_to_rgb: true`,
and it refuses a 3-channel non-RGB processor (`YCbCrProcessor`) unless you set
`allow_non_rgb: true` — feeding Y/Cb/Cr planes to VGG as though they were R/G/B
trains on features that mean nothing. The frozen VGG is deliberately excluded
from checkpoints, so a `.ckpt` trained under one criterion loads into a module
configured with another.

## SRGAN

[`templates/config.srgan.template.yaml`](templates/config.srgan.template.yaml)
reproduces **SRGAN-VGG54**, the paper's headline variant: the SRResNet generator, a
VGG19 `φ_{5,4}` content loss, and `sisr.losses.AdversarialLoss` (non-saturating, over
the critic's logits) weighted `1e-3`, alternating one `sisr.models.srgan.SRDiscriminator`
update per generator update. `sisr.training.SRGANLightning` owns both optimizers under
manual optimization; the YAML selects it by `class_path` on the top-level `model:` block,
which is what that block's `class_path`/`init_args` nesting is for.

```bash
pip install '.[perceptual]'   # LPIPS. DISTS needs no extra.
sisr fit --config templates/config.srgan.template.yaml
```

**The generator starts from an MSE-trained SRResNet.** Ledig et al. scope the
MSE-initialisation trick to "when training the actual GAN", so a paper-faithful run
points `training_config.init_from` at a finished SRResNet run's bare-weights
`.safetensors` —
never the sibling `.ckpt`, which holds the whole LightningModule. That file's own
provenance metadata is checked against this run's generator, processor, output range and
scale, and refused on any mismatch: weights trained under a different one produce a model
that trains and scores without ever erroring. Set `init_from: null` in the YAML to train
from scratch, which is not the paper's recipe (`--...init_from=null` on the command line
does not work — jsonargparse coerces it to the string `'None'`). A dotted CLI override
of one field is safe on `training_config` here but not on `eval_config`, and the
difference is the annotation each argument carries, not the field. `SRGANLightning`
types `training_config` as `SRGANTrainingConfig | None`, so the bare-annotation rebuild
described under
[Config overrides and subclass defaults](#config-overrides-and-subclass-defaults) lands
back on the subclass itself — overriding `adversarial_weight` or `d_steps_per_g_step`
alone costs nothing. `eval_config` is still typed at the base `SREvalConfig | None`,
exactly as it is on `SRLightning`, so a dotted override of one of its fields reverts the
rest: `perceptual_metrics` empties — silently removing the only metric family that
tracks an adversarial objective — and `ssim_impl` flips from `daala` to `wang`, with
nothing logged. YAML, or the whole-object JSON override from that section, is what to
use for any `eval_config` field here.

**`global_step` runs ahead of the batch count here, and only here.** It counts optimizer
steps, and this module takes one discriminator step every batch plus one generator step
every `k`-th batch, so after `N` batches it reads `N + N // k` — twice `N` at the default
`k = 1`. This is unchanged Lightning behaviour, not a quirk of
manual optimization: manual optimization with a *single* optimizer still gives
`global_step == batches`, measured. No other architecture here is affected. Two knobs
that read alike are in different units, so check which before copying a number between
templates:

| Counted in global steps | Counted in batches |
|---|---|
| `trainer.max_steps`, `every_n_train_steps`, the `{step}` in checkpoint filenames | `val_check_interval`, `log_every_n_steps`, LR-scheduler milestones |

`every_n_train_steps` needs an even value to fire at its stated cadence, since
`global_step` only takes even values here — an odd one fires at twice its nominal
period (mechanics in the template's comment).

TensorBoard's default x-axis is the batch counter, so a curve and the checkpoint pulled
off it are a factor of 2 apart.

**PSNR and SSIM get worse by design.** An adversarial objective buys perceptual detail by
spending distortion, which is what those two measure. The template's checkpoints are
therefore rolling rather than monitored — `monitor_metric: null` with `keep_last: 3`
keeps the last three saves by step — because a `save_top_k` on `psnr/val/RGB` or
`ssim/val/RGB` selects the *least* adversarial state of the run, typically one from its
first few thousand steps. `SRWeightsCheckpoint` does the same for distributable weights,
once per network via `attribute: model` and `attribute: discriminator`.

**Perceptual metrics are what track the objective instead.** `SRGANEvalConfig` adds
`lpips` and `dists` on top of SRResNet's scoring, logged as `lpips/val` and `dists/val`
during validation and as `lpips/{set}` / `dists/{set}` per benchmark set. Both are
lower-is-better, so a checkpoint monitoring one needs `mode: min`; `SRCheckpoint` refuses
the wrong direction at setup rather than quietly keeping the worst model of the run.

**MOS is not reproduced.** Ledig et al.'s headline result is a human mean-opinion-score
study, and nothing here stands in for one. LPIPS and DISTS correlate with human judgement
better than PSNR does — that is the entire claim being made for them, not that they are a
substitute.

The template also records a host-RAM OOM in the validation dataloader workers, hit at the
first validation rather than at step 0; read the comment beside its
`val_dataloader_kwargs` before choosing worker counts.

## Reproduction results

Both architectures are scored against the source paper's own tables, under the source
paper's own metric convention. **One reproduces and one does not**, and both are published
here — a table that shows only the win is not evidence of anything.

### SRResNet reproduces Ledig et al.

Ledig et al., ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network"](https://arxiv.org/abs/1609.04802), Table 2, SRResNet row, ×4. Ours is trained on
DIV2K-800; theirs on 350k ImageNet images. Same 1e6 iterations, same 1e-4 learning rate.

| Set | PSNR-Y | paper | Δ | SSIM-Y (daala) | paper | Δ |
|---|---|---|---|---|---|---|
| Set5 | **32.0571** | 32.05 | **+0.007** | **0.9002** | 0.9019 | **−0.0017** |
| Set14 | **28.4899** | 28.49 | **−0.000** | **0.8167** | 0.8184 | **−0.0017** |
| BSD100 | **27.5142** | 27.58 | −0.066 | **0.7588** | 0.7620 | **−0.0032** |

**Read the SSIM column as daala**, the convention Ledig et al. used — see
[Comparability](#comparability) for why that is not interchangeable with the Wang
convention most later papers report. Under Wang the same weights score 0.8916 / 0.7799 /
0.7346, which is a different measurement of the same model and **not** comparable to the
table above.

The residual is small *and uniform* (−0.0017 / −0.0017 / −0.0032) rather than varying
several-fold, which is what a dataset-size offset looks like and what a convention mismatch
does not. Correcting PSNR by a training-free bicubic control — scoring plain bicubic
upsampling through this stack and comparing to the paper's own bicubic row — leaves
+0.008 / −0.098 / −0.081 dB. The honest summary is **within ~0.1 dB of the paper, on 800
images against their 350k**.

### SRGAN-VGG54 does not, and the cause is isolated

Same paper, Table 2, SRGAN-VGG54 row. This run completed the paper's stated schedule
(2×10⁵ update iterations) with every documented hyperparameter matched.

| Set | PSNR-Y | paper | Δ | SSIM-Y (daala) | paper | Δ |
|---|---|---|---|---|---|---|
| Set5 | 25.654 | 29.40 | **−3.75** | 0.7850 | 0.8472 | **−0.0622** |
| Set14 | 23.797 | 26.02 | **−2.22** | 0.6657 | 0.7397 | **−0.0740** |
| BSD100 | 23.065 | 25.16 | **−2.10** | 0.5910 | 0.6688 | **−0.0778** |

**The adversarial pipeline itself is sound.** A single-variable ablation — the same run with
`torch.nn.MSELoss` as the content loss and everything else identical — replicates the
paper's SRGAN-MSE row (Table 1, Set5/Set14 only):

| | ours | paper | Δ |
|---|---|---|---|
| SRGAN-MSE, Set5 | 30.460 / 0.8603 | 30.64 / 0.8701 | −0.18 dB / −0.0098 |
| SRGAN-MSE, Set14 | 27.571 / 0.7791 | 26.92 / 0.7611 | **+0.65 dB / +0.0180** |

So `SRGANLightning`, `SRDiscriminator`, `AdversarialLoss` and the manual-optimization step
order are exercised end to end and land on the paper. The defect is isolated to the
**VGG54 content-loss configuration**.

**The mechanism is a scale-matching constant that does not hold on this data.** The paper
rescales VGG feature maps by 1/12.75 with a stated purpose: *"to obtain VGG losses of a
scale that is comparable to the MSE loss"*. The implementation here is faithful to that
description, but measured on DIV2K-800 the VGG54 content term comes out **5.55× smaller**
than the MSE content term — so the same `adversarial_weight` buys a ~5.5× stronger
adversarial pull, and the paper's stated comparability is not achieved. Tracked in **#120**;
the corrective run has not been made, so no fixed number is claimed here.

### SRCNN has no publishable row

SRCNN is fully wired and trains, but every SRCNN run in this project predates the
MATLAB-`imresize` switch, all three metric-convention fixes, and a correction to the paper's
weight-init standard deviation. Those numbers are not comparable to anything, including to
each other, so none are published. A comparable SRCNN row needs a fresh run.

### Where the papers stop specifying

A claim that these numbers can be independently verified is only honest if the places the
source papers **stop specifying** are stated as plainly as the places they do. Each of the
following is a choice this implementation had to make and **cannot be scored as correct**,
because there is nothing to be faithful to. None is a bug, and none should be "fixed" by
picking a value and asserting it in code — that would dress a guess as a checked fact.

| choice | what ships | what it rests on |
|---|---|---|
| SRCNN batch size | `64` | **Nothing.** Not in the paper, not on the authors' project page, re-verified twice. The authors' released Caffe solver prototxt would settle it and **has not been inspected**; a community reimplementation is not a substitute. |
| SRCNN training budget | `max_steps: 10000000`, ≈0.8× the paper's figure | The paper gives *"the same number of backpropagations (i.e. 8×10⁸)"*. Read here as **per-sample** updates, because the mini-batch reading implies an implausible iterations-per-second rate on 2014 hardware. That is an inference. An earlier reading of ours recorded *"up to 8×10⁸"* — a ceiling rather than a target — and the two readings are **unreconciled**. |
| SRCNN evaluation border | `crop_border: 3`, on top of the 6 px per side that `padding: valid` already trims | The paper states no test-time boundary convention. Neither value is correct or incorrect, only stated. |
| SRResNet PReLU parameterisation | `torch.nn.PReLU()` — one shared slope | The paper specifies neither shared nor per-channel, and no official reference implementation exists to check against. |

The practical consequence: a reproduction that differs from this project on any of these
rows is not necessarily wrong, and neither is this one. Compare the rows before comparing
the numbers.

### What these were computed against

- **Weights**: the reference SRResNet run at step 1,000,000 (converged; plateau from ~500k)
  and the adversarial runs at 200,000 batches. Rescored through the `test` path rather than
  read from training logs, with PSNR reproducing to ~1e-4 dB as the control that the
  rescoring path agrees with the training path.
- **Training data**: DIV2K-800 HR, LR derived at load by the vendored MATLAB `imresize`.
- **Benchmark data**: the EDSR authors' benchmark distribution — the exact archive, its
  SHA-256 and this project's Set14 variant are pinned under [Comparability](#comparability).
- **Metrics**: Y-channel, BT.601 studio range, output clamped to `[0, 1]`, per-image PSNR
  averaged over the set (not pooled-MSE). All four choices are consequential and all four
  are justified under [Comparability](#comparability).

## Comparability

Benchmark numbers only mean something if the inputs and the metrics match the papers', so
both are pinned:

- **LR generation** uses a vendored MATLAB-compatible `imresize`
  ([`sisr/utils/imresize.py`](sisr/utils/imresize.py), MIT, attribution in the file header) rather than
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
  only reconstruction quality. [`sisr/metrics/ssim.py`](sisr/metrics/ssim.py) ports daala's method,
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
- **An LPIPS figure is comparable only to one computed under the same backbone**, and
  the discipline is the same as SSIM's. `SREvalConfig.lpips_net` selects `'alex'` (the
  default, and what the SR literature usually reports), `'vgg'` or `'squeeze'`; these
  are three different learned networks, not three reductions of one number, so they do
  not agree on the same image. As with `ssim_impl`, the tag (`lpips/val`) and the
  checkpoint filename carry the bare value and nothing about how it was produced — the
  backbone is recorded in `hparams` and in every artifact's `sisr_meta`. DISTS has no
  such knob.
- **These are not idiosyncrasies of this project — two peer-reviewed surveys document
  the same field-wide inconsistencies.** Keleş, Yılmaz, Tekalp, Korkmaz and Doğan,
  ["On the Computation of PSNR for a Set of Images or Video"](https://arxiv.org/abs/2104.14868)
  (Picture Coding Symposium 2021, arXiv:2104.14868), find no agreed convention for
  aggregating PSNR across a set of images — arithmetic mean of independently-computed
  per-image PSNR values versus a single PSNR from pooled MSE — with the two diverging by
  up to ~2.5 dB on the same data. This project uses the former (their convention (a), not
  MSE-pooling): PSNR is computed one image at a time and the per-image values are then
  averaged (`sisr/training/callbacks.py:365-370` computes each image's PSNR, `:440-442`
  takes the arithmetic mean over them), independently confirmed by
  [`SRLightning._mean_psnr`](sisr/training/lightning_module.py)
  (`sisr/training/lightning_module.py:504-515`), whose own docstring names and rejects
  MSE-pooling as the alternative. Wang, Chen and Hoi's
  ["Deep Learning for Image Super-resolution: A Survey"](https://arxiv.org/abs/1902.06068)
  (IEEE TPAMI 2020, arXiv:1902.06068, §II-D "Operating Channels") likewise finds no
  accepted best practice for which color space or channels to score SR on, with reported
  results differing by up to 44 dB depending on the choice — the same studio-range-vs-full
  and Y-vs-RGB distinctions this project names and pins above.
- **Benchmark images are the EDSR authors' own benchmark distribution.** The Set5/Set14/
  BSD100 HR images and MATLAB-`imresize`-generated LR pairs behind every PSNR/SSIM figure
  in this project come from Lim et al., "Enhanced Deep Residual Networks for Single Image
  Super-Resolution" (CVPRW 2017), downloaded from
  `https://cv.snu.ac.kr/research/EDSR/benchmark.tar`, SHA-256
  `80c21c333bbf6ceb5308b7243761f8284478274413a97b96f1d63e9045fd93e8` (recorded and checked
  in [`tests/utils/test_imresize.py`](tests/utils/test_imresize.py)). This project's Set14 is the full
  14-image variant from that distribution — published SR papers' "Set14" numbers have been
  reported over 11-, 12- and 14-image subsets depending on source, so this count is worth
  stating explicitly for anyone comparing numbers against this project's own.
  **Urban100 and Manga109 are not used**, so no figure here is over either — a table that
  reports them is measuring something this project has not measured.
- **The upscale leg has its own reference data, generated rather than downloaded.** The
  distribution above ships HR and MATLAB-`imresize` LR pairs, which covers the downscale
  leg only. SRCNN's degradation is bicubic-down *then* bicubic-up, so the second leg is
  verified against `Bicubic_up` references generated in MATLAB from those same LR images —
  the exact expression, sizes and directory layout are recorded in
  [`tests/utils/test_imresize.py`](tests/utils/test_imresize.py)'s module docstring, so the
  data can be regenerated rather than trusted. Byte-equality currently holds on both legs,
  21 cases, none skipped. The tests skip cleanly when the reference data is absent, which
  keeps CI hermetic — **a skip there means a leg was not exercised, not that it passed**.
- **`pyiqa` (IQA-PyTorch) does not default to these conventions.** Its PSNR metric
  defaults to full RGB (`test_y_channel=False`), and its SSIM defaults to Y-channel but in
  full-range YIQ (`color_space='yiq'`) — not the studio-range BT.601 YCbCr that MATLAB's
  own `rgb2ycbcr.m` and this project's Y-channel figures use. To reproduce this project's
  Y-channel PSNR/SSIM using `pyiqa`, pass `color_space='ycbcr'` explicitly; the library's
  own default will not match. Which of this project's figures are checkable with it:

  | this project's figure | `pyiqa` equivalent | matches out of the box? |
  |---|---|---|
  | `psnr/*/RGB` | `psnr`, default arguments | yes |
  | `psnr/*/Y` | `psnr` with `test_y_channel=True`, `color_space='ycbcr'` | **no** — defaults to full RGB |
  | `ssim/*/Y` at `ssim_impl: 'wang'` | `ssim` with `color_space='ycbcr'` | **no** — defaults to full-range YIQ |
  | `ssim/*/Y` at `ssim_impl: 'daala'` | *none* | **not reproducible** — no fixed-window SSIM implements daala's height-scaled sigma |
  | `lpips/*` | `lpips` with the matching backbone | only if the backbone matches; see above |
  | `dists/*` | `dists`, default arguments | yes |

  **The daala row is the one that matters most**, because it is the convention every
  SRResNet figure in this project is reported under. A `pyiqa` SSIM cannot check those
  numbers, only the `'wang'` ones.

## Config overrides and subclass defaults

**`--ckpt_path` preserves subclass identity.** `SRLightning` saves `eval_config` in
`hparams` as `dataclasses.asdict(self.eval_config)` — a bare dict with no `class_path`
([`sisr/training/lightning_module.py`](sisr/training/lightning_module.py)) —
and `SRLightningCLI._parse_ckpt_path` merges that dict, through the *subcommand's own*
parser, onto the class `--config` has already selected
([`sisr/cli.py`](sisr/cli.py)). The stored keys land as `init_args` overrides on
`SRResNetEvalConfig`; every key the dict does not mention keeps the subclass's own
default, not the base class's. So a checkpoint saved before `ssim_impl` existed still
reloads as an `SRResNetEvalConfig` scoring `daala`, and `crop_border=4` /
`psnr_channels=['RGB', 'Y']` survive because the class does, not because
`dataclasses.asdict` happened to record them. This holds on every subcommand that
accepts `--ckpt_path`, `fit` resume included, and is locked by
`test_ckpt_path_preserves_eval_config_subclass_identity` in
[`tests/test_cli.py`](tests/test_cli.py), which asserts both the reloaded object's class
and a subclass-only default absent from the checkpoint's stored dict. Earlier versions
of this README described patching a copy of a checkpoint's `hyper_parameters` before
re-scoring; that recipe existed for this failure only, and is obsolete.

**A dotted command-line override used to reset the config it touched — fixed.** This was a
separate defect from the checkpoint issue above, with a separate cause: jsonargparse treats a
pure dataclass as a *closed* type by default, so setting one field of a subclass-typed config
from the CLI rebuilt the whole object from its **bare annotation**, and every *other* default
the subclass had overridden silently reverted with it. `SRLightningCLI.__init__` now calls
`set_parsing_settings(subclasses_enabled=[SREvalConfig, SRTrainingConfig])` before the parser
builds, so a dotted override applies its field without rebuilding the rest:

```bash
sisr validate --config templates/config.srresnet.template.yaml \
              --model.init_args.eval_config.ssim_impl=wang
```

correctly keeps `eval_config` as `SRResNetEvalConfig`. `ssim_impl` is `wang` as asked, and
`crop_border` stays `4` and `psnr_channels` stays `['RGB', 'Y']` — nothing else reverts. The
same now holds for `training_config`. Naming the class in a separate argument was never
necessary and still isn't; the dotted override applies cleanly on its own.

Two forms also work, and remain useful when overriding more than one field at once. Set the
field in YAML — in the config itself, or in an overlay passed as a second `--config`:

```yaml
model:
  init_args:
    eval_config:
      init_args:
        ssim_impl: wang
```

or pass the whole object as one JSON argument, `class_path` included:

```bash
sisr validate --config templates/config.srresnet.template.yaml \
  --model.init_args.eval_config='{"class_path": "sisr.models.srresnet.SRResNetEvalConfig", "init_args": {"ssim_impl": "wang"}}'
```

`--print_config` confirms the subclass survived either way: the resolved `eval_config` shows a
`class_path` when the subclass survived, and a bare key/value mapping when it did not.

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

## Model weights

**No pretrained checkpoint is released yet.** This section is the policy that will apply
when one is, written now because it is free to decide today and awkward to decide under
pressure later.

**Licence: MIT, matching the code**, with the caveat below stated alongside the download
rather than buried.

**The caveat.** Training data such as DIV2K carries an *"academic research purpose only"*
restriction. Whether that kind of training-data restriction legally propagates to a
derivative model's weights is a **genuinely unsettled question** in ML/copyright practice
today. A prospective commercial user of a future checkpoint should treat it as an open
question, not a resolved one — this project states the input and declines to assert the
conclusion in either direction.

**What this repository redistributes, now and at release.**

| artifact | redistributed? |
|---|---|
| Source code | Yes — MIT. |
| Training data (DIV2K) | **No.** Fetched by the user; `data/` is not tracked. |
| Benchmark data (Set5, Set14, BSD100) | **No.** Fetched by the user from the EDSR authors' archive, whose SHA-256 is pinned under [Comparability](#comparability) so the copy can be checked without being shipped. |
| Generated LR / `Bicubic_up` reference pairs | **No** — they are derivatives of the above and inherit whatever those carry. |
| Trained weights | Not yet. MIT when released, under the caveat above. |
| Benchmark output images (SR reconstructions) | Not yet, and **not without a per-set check first**. |

**Benchmark sets carry real and differing licensing friction**, which is why the last row is
not a blanket yes. Set5, Set14 and BSD100 originate from separate publications with separate
terms, and "we redistribute nothing" is what currently makes that moot. Any release that
ships reconstructed images — a qualitative figure, a demo gallery — stops it being moot and
needs the terms of that specific set checked first. Pinning the archive hash rather than
mirroring the archive is a deliberate choice in the same direction.

**The tests are hermetic for this reason too.** Every test that needs benchmark data skips
cleanly when it is absent, so CI never requires a copy and a contributor without the archive
still gets a green suite. **A skip means the case was not exercised, never that it passed**
— check the skip count, not just the exit code.

## License

MIT, see [LICENSE](LICENSE). All dependencies are MIT-compatible.
