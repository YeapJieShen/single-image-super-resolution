# Reproduction reference

Everything about **whether the numbers reproduce, and under exactly what convention**. Split out
of the README on 2026-09-01 (#198), because this is reference material a reader consults with a
specific question rather than narrative anyone reads top to bottom, and it was the section that
kept growing.

**This file is the single owner of these figures and conventions.** The README carries the
headline numbers and links here; it does not restate the tables. If you are adding a result,
add it here.

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
source papers **stop specifying** are stated as plainly as the places they do. Such a choice
**cannot be scored as correct**, because there is nothing to be faithful to. It is not a bug,
and it should not be "fixed" by picking a value and asserting it in code — that would dress a
guess as a checked fact.

**This table held four rows until 2026-09-01. One remains.** What happened to the other three
is recorded below, because a search that closed a row is worth as much as the row was.

| choice | what ships | what it rests on |
|---|---|---|
| SRResNet PReLU parameterisation | `torch.nn.PReLU()` — one shared slope | The paper specifies neither shared nor per-channel, and no official reference implementation exists. **Narrowed, not settled** — see below. |

The practical consequence: a reproduction that differs from this project on this row is not
necessarily wrong, and neither is this one. Compare the row before comparing the numbers.

#### Three rows left this table on 2026-09-01

SRCNN's **batch size** and **training budget** were listed here as resting on nothing
checkable. They are now specified by a primary source, so they no longer belong in a list of
choices with nothing to be faithful to.

The SRCNN authors released a Caffe training package (`SRCNN_train.zip`, from the project page
both papers link; retrieved 2026-09-01, SHA-256
`001146419f7acfb12a3e7929c8acd5de88a08d687d6881085f81321ad6982b1a`). Its two files settle
both rows at once:

| field | authors' value | what this project now ships |
|---|---|---|
| `hdf5_data_param.batch_size` (train) | `128` | `batch_size: 128` (was `64`) |
| `max_iter` | `15000000` | see below |
| `base_lr` / `momentum` / `weight_decay` | `0.0001` / `0.9` / `0` | already matched |
| per-layer `lr_mult` (weights) | `1`, `1`, `0.1` | already matched |
| per-layer `lr_mult` (biases) | `0.1`, `0.1`, `0.1` | **now matched** — see below |
| `weight_filler` std | `0.001` | already matched |

The **bias** row was not matched, and the same file is what exposed it. Caffe applies a
layer's two `param` blocks to its blobs in order — weights, then bias — so conv1 and conv2 pair
`lr_mult` 1 with 0.1 and every bias trains at a tenth of `base_lr`, independently of the
per-layer weight schedule. This project assigned one rate to a layer's weight *and* bias, so
conv1 and conv2 biases trained at 10x the authors' rate; conv3 happened to agree, because its
weight rate is already 1e-5. A `layer_lrs` entry may now be a `[weight_lr, bias_lr]` pair, and
the shipped SRCNN recipe uses one per layer.

`max_iter` settles one thing and opens another.

**Settled:** the paper's *"the same number of backpropagations (i.e. 8×10⁸)"* **cannot** mean
iterations — 8×10⁸ exceeds the authors' own cap of 1.5×10⁷ by two orders of magnitude. Per-sample
is the only consistent reading, which this project had already inferred and no longer has to.

**Open, and deliberately left open:** at batch 128 the authors' `max_iter` is
15,000,000 × 128 = **1.92×10⁹** backpropagations, against the paper's stated **8×10⁸** — which
would be 6,250,000 iterations. **The released code and the paper disagree, by 2.4×.**

This project ships the authors' value, `max_steps: 15000000`, because that is the number a
primary source states. Choosing 6,250,000 instead would make the paper's sentence come out exact
at the cost of shipping a value no source states, resolved by our inference that `max_iter` is a
ceiling rather than a target. Their `snapshot: 500` makes that inference plausible — but **which
snapshot produced the published numbers is recorded nowhere**, so it stays an inference, and the
honest position is to follow the file and say the two disagree.

Practically this is a long run; stop at a checkpoint rather than lower the config to an
unsourced number.

No published number changes: SRCNN has no publishable row.

**The evaluation border also left**, for the opposite reason: it turned out to be specified
after all, and we differed from it. The authors' demo package (`SRCNN_v1.zip`, retrieved
2026-09-01, SHA-256 `bfa68ca613c1326a59e0c34353205a254ab2b67e34df7f04e28eef567980af30`)
shaves **`scale` pixels per side** before computing PSNR — not a constant — and runs the
network with **`same` padding and replicated borders** at test time, though it trains with
`pad: 0`. This project used `padding: valid` at inference and then `crop_border: 3`, so at ×3
it scored a region strictly *inside* theirs, and the constant agreed with `scale` only at ×3.

The two differences were entangled — changing the border alone moves *away* from the authors,
not toward them, because the valid-convolution loss stays underneath — so both are now
adopted together:

| | authors | ships now |
|---|---|---|
| inference padding | `same`, replicated border | `eval_padding: same`, `eval_padding_mode: replicate` |
| training padding | `pad: 0` | `padding: 0`, unchanged |
| border shaved before PSNR | `scale` px/side | `crop_border` derived from `scale` |

The replicated border is re-derived at **every** convolution, as the authors' per-layer
`imfilter` does; pre-padding the input once and convolving valid is a different function.
`eval_padding` applies only outside training mode, so the model is deliberately not the same
function in the two modes — which is what the authors' own setup is.

The trade this makes is real and was decided rather than assumed: same-padded inference scores
pixels whose receptive field includes invented (replicated) content. Keeping valid convolution
never does, and is arguably more honest, but then the numbers are not comparable to the paper's
on a metric where border handling is known to matter. Reproducing the paper won. Deleting the
two `eval_padding` lines from the config restores valid convolution at train and test.

#### What did *not* settle: SRResNet's PReLU

The parameter count was the only route that could have settled this from the paper alone, and
it is now closed with a measurement rather than an assumption. This project's SRResNet has
**19 PReLU sites, all at 64 channels**:

| | PReLU params | model total |
|---|---:|---:|
| one shared slope (ships) | 19 | 1,549,462 |
| per-channel | 1,216 | 1,550,659 |

The two differ by **1,197 parameters, 0.0773% of the model** — invisible at any precision a
paper reports a parameter count to. No count Ledig et al. could have published would
distinguish them.

The weaker route also fails to decide it: the PReLU paper itself presents channel-wise and
channel-shared as both viable and reports them performing comparably, so "what the authors of
PReLU would have assumed" has no single answer either. The row stays labelled, and the search
is recorded so nobody repeats it.

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
