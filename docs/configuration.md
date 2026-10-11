# Configuration reference

One YAML file describes one experiment. Every architecture runs through the same
`LightningCLI` entrypoint, so the file is the only thing that changes:

```bash
sisr fit -c templates/config.srresnet.template.yaml
```

Copy a template out of [`templates/`](../templates/) and point its dataset paths at
wherever your images live. This page explains the settings; the templates stay
close to bare YAML so they read as configuration rather than as prose.

> **Config files must stay ASCII.** jsonargparse opens them with the platform
> encoding, so a non-ASCII character is a hard `UnicodeDecodeError` at launch.

---

## Top-level keys

| Key | What it does |
| --- | --- |
| `seed_everything` | Seeds Python, NumPy and torch. |
| `matmul_precision` | Sets `torch.set_float32_matmul_precision`. |
| `optimizer` | The optimizer. For SRGAN this is the **generator's**. |
| `lr_scheduler` | Optional schedule. For SRGAN this is the **generator's**. |
| `model` | The Lightning module and everything it wraps. |
| `data` | Datasets and dataloader settings. |
| `trainer` | Schedule, precision, callbacks, loggers. |

### `matmul_precision`

`high` enables TF32 matmul on Ampere and later. Measured a **no-op on this repo's
conv stacks** — convolutions run through cuDNN, whose `allow_tf32` already defaults
to true. It is kept for matmul-heavy architectures added later.

---

## The `model` block

`model.class_path` selects the Lightning module: `sisr.training.SRLightning` for
ordinary supervised training, `sisr.training.SRGANLightning` for adversarial.
Everything else nests under `model.init_args`.

### `model` (the network) and `processor`

The processor is the colorspace adapter between the data and the network:

| Processor | Model sees | Used by |
| --- | --- | --- |
| `RGBProcessor` | RGB in `[0, 1]` | EDSR, MDSR, ESRGAN, ESPCN |
| `RGBSignedOutputProcessor` | LR `[0, 1]` in, HR/target `[-1, 1]` | SRResNet, SRGAN, ESRGAN |
| `YChannelProcessor` | Full-range Y channel only | SRCNN |
| `YCbCrProcessor` | YCbCr in `[0, 1]` | no shipped architecture; for a 3-channel non-RGB model |
| `StudioYChannelProcessor` | MATLAB studio-range Y (16-235) only | VDSR, DRCN |

`RGBSignedOutputProcessor` is the range Ledig et al. use (§3.2). **Changing the
processor renumbers what the network is trained on**, and under SRGAN it also
changes the space the discriminator scores in.

### `criterion`

Unset means `torch.nn.MSELoss`, which is what both papers' baselines use. Any
`nn.Module` taking `(pred, target)` works, so L1 needs no project class:

```yaml
criterion: {class_path: torch.nn.L1Loss}
criterion: {class_path: sisr.losses.CharbonnierLoss}
```

**Ledig et al.'s SRResNet-VGG22**, faithfully: a VGG22 content loss plus total
variation at `2e-8`, with **no pixel term**, trained from scratch on the schedule
the template already configures. The MSE-initialisation trick in the paper applies
to the SRGAN variants, not to this one.

```yaml
criterion:
  class_path: sisr.losses.WeightedSumLoss
  init_args:
    terms:
      vgg22:
        class_path: sisr.losses.VGG19FeatureLoss
        init_args: {layer: vgg22}
      tv:
        class_path: sisr.losses.TotalVariationLoss
    weights: {vgg22: 1.0, tv: 2.0e-8}
```

Two caveats before running it: the first use downloads **~548 MB of VGG19 weights**
to the torch hub cache, and TV's `2e-8` presumes the `[-1, 1]` output range that
`RGBSignedOutputProcessor` provides.

| Class | What it is |
| --- | --- |
| `torch.nn.MSELoss` | the default; both papers' baseline |
| `torch.nn.L1Loss` | plain L1 |
| `sisr.losses.CharbonnierLoss` | `sqrt(diff² + eps²)`: L1 with a finite gradient at zero |
| `sisr.losses.TotalVariationLoss` | isotropic TV regulariser; ignores its target |
| `sisr.losses.VGG19FeatureLoss` | VGG19 feature-space perceptual loss, any `φ_{i,j}` |
| `sisr.losses.VGG16FeatureLoss` | the same on VGG16 (deepest is `vgg53`); experimentation only, no published SR recipe uses it |
| `sisr.losses.WeightedSumLoss` | named weighted sum of any of the above |

A `WeightedSumLoss` logs each term separately as `loss/train/{name}` and `loss/val/{name}`, so
you can see which one dominates. Its `weights` may be scheduled per term with
`sisr.training.callbacks.LossWeightScheduler`.

A VGG loss normalises with RGB ImageNet statistics, so it refuses a 1-channel processor
(`YChannelProcessor`) unless you set `grayscale_to_rgb: true`, and a 3-channel non-RGB one
(`YCbCrProcessor`) unless you set `allow_non_rgb: true`: Y/Cb/Cr planes fed to VGG as R/G/B
train on features that mean nothing. The frozen VGG is excluded from checkpoints, so a `.ckpt`
trained under one criterion loads into a module configured with another. `weights: null` skips
the download and builds a random VGG that warns; it is for tests only.

**Choosing `layer` (φ<sub>i,j</sub>)**: blocks 1–2 have 2 convs at both depths
(`vgg11`–`vgg22`); blocks 3–5 have 4 on VGG19 (up to `vgg54`) or 3 on VGG16 (up to
`vgg53`). `before_activation: true` selects ESRGAN's pre-ReLU features.

> A perceptual run scores **worse PSNR by design**. The templates checkpoint on
> `ssim/val/RGB` as well as PSNR, so the PSNR-monitored "best" stops tracking the
> training objective — decide which monitored checkpoint you actually want.

### `training_config`

| Field | Notes |
| --- | --- |
| `example_input_shape` | **Optional.** Derived from the real training patch when unset; checked against it when set. |
| `compile_backend` | A `torch._dynamo` backend name, or unset for eager. |
| `layer_lrs` | Per-`Conv2d` learning rates; requires every trainable parameter to live in a `Conv2d`. |
| `init_strategy`, `init_mean`, `init_std` | Weight initialisation. |
| `scale` | Upscaling factor. **Required whenever a run writes an artifact and the model has no `scale` of its own.** |

`scale` is stated here for SRCNN and inherited from the network for SRResNet, and that
asymmetry is real rather than an inconsistency to tidy away. SRResNet upsamples internally, so
its scale is a property of the network and lives in its hyperparameters. SRCNN is
resolution-preserving — it refines an input that has *already* been upsampled — so its scale is
a property of the **data**, and nothing about the network records it.

Set it anyway. Saving an artifact whose scale cannot be resolved from either source is refused
rather than recorded as null: a file that cannot state its factor is unusable by anyone who did
not train it, because for a pre-upsampled architecture the factor is exactly what they must
resize the input by before feeding it. It is deliberately *not* inferred from the datamodule —
`scale` is not part of the dataset contract, and a predict-only dataset has none at all.

`example_input_shape` is **not** only a TensorBoard-graph dummy. It also shapes the
`torch.compile` warm-up — a wrong size compiles once and then recompiles on step 1 —
and it drives `ModelSummary`'s FLOPs count.

You may leave it unset: it is derived from the first real training patch. Setting it is
still useful, because a stated value is *checked* against that patch rather than
replaced by it — an intent that can be contradicted. **The shipped templates state it**
for one reason: they are also the input to `sisr export`, which never sees training data
and so has nothing to derive from.

### `eval_config` and the two SSIM conventions

**A metric figure is only comparable to one computed the same way.** SSIM has two
live conventions here, and `ssim_impl` switches between them:

| `ssim_impl` | Method | Comparable to |
| --- | --- | --- |
| `daala` | σ scales with image height | Ledig et al.'s SRResNet/SRGAN paper |
| `wang` | fixed 11×11 gaussian, σ 1.5 | torchmetrics, MATLAB, BasicSR, EDSR/RCAN/SwinIR |

`SRResNetEvalConfig` and `SRGANEvalConfig` default to **`daala`**, because that is
what their paper used. `SRCNNEvalConfig` defaults to **`wang`**, the field standard.
Override explicitly when you need the other:

```yaml
eval_config:
  class_path: sisr.models.srresnet.SRResNetEvalConfig
  init_args: {ssim_impl: wang}
```

Other fields: `crop_border`, `psnr_channels`, `ssim_channels`, `separate_psnr`,
`perceptual_metrics` (the paired `lpips` and `dists`, and the no-reference `niqe`,
`ma` and `pi`; all but `dists` need the `[perceptual]` extra, and `pi` applies PIRM's
own 4-pixel shave on top of `crop_border`), and `lpips_net` — a LPIPS figure is only
comparable under the same backbone.

**Paper protocols.** Four independent switches reproduce what a paper's MATLAB does; each
paper's template sets its own, and all default off, so a config that names none scores exactly as
before:

| Switch | Where | What it does |
| --- | --- | --- |
| `round_to_uint8` | `eval_config` | Rounds each scored plane (SR and HR) to 8-bit levels before PSNR; SSIM and perceptual metrics read the unrounded tensors. |
| `grey_plane_scoring` | `eval_config` | Scores a single-channel image *file* on its raw plane instead of studio-range Y. Greyness is the file mode, reported per image by the eval dataset's `yield_is_grey: true`; it is never inferred from pixel values (Set14's `man` is an RGB file with equal channels and is colour to MATLAB). |
| `float_degrade` | `pre_upsampled.ValidationDataset` | Degrades the LR input in float, with no uint8 re-quantization between the down and up passes. |
| `clamp_input_luma` | `pre_upsampled.ValidationDataset` | Clamps a colour image's network-input luma to `[16, 235]/255`, as `VDSR.m` and `DRCN.m` do. Evaluation only; training is untouched. |

A figure is comparable only to one computed under the same switches.

---

### Multi-scale models and self-ensemble

A multi-scale model (`VDSR`, `MDSR`, any model with `multi_scale = True`) trains one set of
weights for several factors: set `training_config.scale: [2, 3, 4]` and a scale is drawn per patch
or batch. Validation then runs once per scale and logs `psnr/val/Y/x2`, `psnr/val/Y/x3`, ...;
the plain tags (`psnr/val/Y`) are the **mean over the scales**, which is a monitor for
checkpointing and not a paper number. `eval_config.predict_scale` selects the scale `sisr predict`
runs at, and is required when a multi-scale model has several scales.

`eval_config.self_ensemble: true` averages the model's output over its eight flip/rotation
variants at **test and predict time only**, never during training or fit-time validation. It costs
eight forward passes and is what makes EDSR+ and MDSR+.

## The `data` block

Datasets are `{class_path, init_args}` specs, instantiated lazily so an expensive
constructor only runs for the stages that need it.

**LR is derived at load** via a MATLAB-compatible antialiased bicubic resize
([`sisr/utils/imresize.py`](../sisr/utils/imresize.py)), byte-exact against MATLAB's
own `imresize`, which is what makes these numbers comparable to published ones.

`test_datasets` are surfaced through both `val_dataloader` (so they are scored during
`fit` for monitoring) and `test_dataloader` (for `sisr test --ckpt_path`).

Training scales must be odd or a power of two for the pre-upsampled train dataset (x6 raises).

> A dotted CLI override cannot reach inside a dataset spec's `init_args` — the field
> is an opaque dict to jsonargparse, so the override lands as a stray sibling key.
> The datamodule refuses that rather than silently ignoring it. Edit the YAML.

### Dataloader workers

**Workers are load-bearing on real data**, because deriving LR per sample is a real
cost and `num_workers=0` hits a hard serial floor. The one-time worker spawn cost
amortizes to nothing over a full run.

Measured on real DIV2K, steady state:

| Config | `num_workers=0` | `num_workers=16` |
| --- | --- | --- |
| SRResNet, batch 16 | 56.0 ms/step | 40.1 ms/step |
| SRCNN, batch 64 | 79.3 ms/step | 14.9 ms/step |

> **The SRGAN template is more exposed than the others on the same data path.** A
> resident discriminator plus a VGG19 feature extractor consume RAM the SRResNet
> template does not need. The template's counts (16 train / 2 val) have **OOM'd host
> RAM on a 16-worker box** — a val worker's float64 resize of a full-size DIV2K image
> raised `numpy._ArrayMemoryError` at the *first validation*, not at step 0. A smoke
> run only completed at 4 train / 0 val workers. Safe counts on any given box are
> unmeasured; lower these if you OOM.

---

## The `trainer` block

### `benchmark`

cuDNN autotunes a convolution algorithm per input shape and caches it — free here,
since every training step feeds the same fixed crop shape. Lightning ties this key to
`deterministic`, which is why it lives in the trainer block rather than as a custom key.

### Precision and compilation

The templates train **eager fp32**, which is bit-identical by definition and what a
paper reproduction needs.

**`bf16-mixed` on its own is a regression, not a speed-up**, measured twice on two
platforms (0.72× on Windows; 41.0 vs 35.5 ms/step eager on Linux, 0.87×). Autocast's
cast-churn does not saturate the SMs at these batch sizes. Crossover was measured at
batch 32, where bf16 with a fused optimizer wins — revisit only if `batch_size` rises.

It is a different story combined with `torch.compile`'s inductor backend, which fuses
those casts into the kernels. That is an opt-in path and it is **both settings
together**:

```yaml
trainer:
  precision: bf16-mixed
model:
  init_args:
    training_config:
      init_args:
        compile_backend: inductor
```

Two things to know before using it for anything you intend to compare against a paper:

- **The PSNR/SSIM cost of bf16 has not been measured.** Stay in fp32 for reproduction
  runs until a fidelity comparison exists.
- **Inductor needs a C compiler at runtime**, not just the `triton` package — it shells
  out to build each kernel. A box with no toolchain fails at the first compiled step
  with `RuntimeError: Failed to find C compiler`.

### 🚨 Step axis: two counters, different units

`trainer.global_step` counts **optimizer steps**. Every `self.log` metric instead
lands on Lightning's `_batches_that_stepped`, which counts **batches**.

For `SRLightning` the two differ only by one. **For `SRGANLightning` they differ by a
factor of two**, because it takes two optimizer steps per batch (one discriminator,
one generator) under manual optimization; more exactly, with `k = d_steps_per_g_step` it reads `N + N // k` after `N` batches. This is unchanged Lightning behaviour, not a
quirk of manual optimization — measured, manual optimization with a *single* optimizer
still gives `global_step == batches`.

So the units differ, and you must read this before changing any of them:

| In **global steps** | In **batches** |
| --- | --- |
| `max_steps` | `val_check_interval` |
| `every_n_train_steps` | `log_every_n_steps` |
| | checkpoint filenames and metadata |
| | hand-stepped LR scheduler `milestones` |

`_batches_that_stepped` is also the **default x-axis every logged metric is plotted
against**, which is why checkpoints are stamped with it: a saved file exists to be
located on a curve, and one named in optimizer steps cannot be. So
`SRResNet_x4_RGB_16B64F_s10000.safetensors` is the state at TensorBoard x=10000 under every
paradigm. The artifact's own metadata records **both** counters under distinct names —
`global_step` keeps meaning the optimizer count, `batch_step` is the axis above — so a
reader that only knows the older field still reads a true value.

Note the asymmetry in the table: `every_n_train_steps` is Lightning's own and counts
optimizer steps, while the filename it produces counts batches. Under SRGAN that means
a cadence of 10000 writes a file named for batch 5000.

### Multi-GPU

`SRLightning` runs distributed. Two settings are required together:

```yaml
trainer:
  devices: 2
  strategy: ddp
  use_distributed_sampler: false     # required -- see below
```

**`use_distributed_sampler: false` is not optional, and a distributed run refuses to
start without it.** This project shards the *training* loader itself and leaves the
validation and benchmark loaders whole on every process. Lightning's injection is a
single trainer-wide switch, so leaving it on would split those too — and
`DistributedSampler` pads a set to a multiple of the world size by **repeating** samples.
A five-image benchmark across four processes becomes eight, three counted twice, and the
reported figure is no longer that set's score. Every process therefore evaluates every
benchmark image; the cross-process reduction on those metrics is then exact, and doubles
as a check that the processes have not diverged.

Every validation metric is reduced across processes. Without that each process reports
its own value as though it were global, and the checkpoint monitors read those and pick
different checkpoints on different processes. The bare-weights artifact is written by one
process only.

**`SRGANLightning` still refuses multi-GPU.** Manual optimization opts out of Lightning's
gradient synchronisation, so the two networks' gradients would not be reduced and each
process would train its own divergent pair with nothing failing. Supporting it means
writing that synchronisation explicitly; that is separate work.

**What has been proven, and what has not.** The rank-correctness above is verified on two
real processes over the gloo backend on CPU, which is what the test suite runs. NCCL
behaviour, device placement on real GPUs and multi-GPU throughput are **not** covered —
those need genuine multi-GPU hardware, and containers do not substitute since they share
the host's devices.

### The distributable artifact

`SRWeightsCheckpoint` writes **safetensors**, and only that. The resumable `.ckpt` is unaffected: Lightning owns that
file, and it carries optimizer moments, loop state and hyperparameters that a flat
tensor map cannot represent.

The reason is not that our own loading was unsafe. Every `torch.load` here passes
`weights_only=True` and the torch floor is pinned for it. The exposure is a *consumer*
opening a published file with that turned off, or on an older torch; safetensors removes
it by construction. Note the consequence: **this makes the file you hand out
pickle-free, not the repository** — a `.ckpt` is still a pickle.

Provenance travels in the file's header, one entry per top-level field rather than one
opaque blob, so anything that can open the artifact gets a readable table. The ONNX
export writes the same fields the same way.

Filenames say what the model is, so a directory of weights can be read without
opening anything:

```
SRCNN_x2_Y_915_s500000.safetensors
SRResNet_x4_RGB_16B64F_s500000.safetensors
SRDiscriminator_96_s500000.safetensors
```

Architecture, scale, colourspace, variant, step. The **variant** is
architecture-specific — SRCNN's kernel triple, SRResNet's block and filter counts — and
each architecture supplies its own via `variant_tag`; no generic rule over
hyperparameters produces a readable tag for all of them. A component drops scale and
colourspace, since neither describes a critic.

The name is **derived from the artifact's own provenance**, not assembled separately, so
a file whose name and header disagree is not representable. Set `filename_prefix` to
override it for a run that wants its own naming.

Two things are deliberately absent. There is **no metric token**: runs monitor different
metrics and adversarial runs monitor none, so it would make otherwise-identical files
look unrelated — the value is in the header instead. And `s<step>` is the **batch** axis,
which is the axis every logged metric uses, so the number can be found on a curve.

Mismatches are checked on read. Fields that change what the output *means* — the
processor and the output range — are refused, because being wrong about either produces
a plausible image and no error. Library-version drift is warned about and loaded, since
it cannot change what the tensors mean and refusing would expire every artifact on the
next dependency bump.

One further trap: `every_n_train_steps`' condition is checked once per batch, so an
**even** value halves the batch cadence (10000 → every 5000 batches) while an odd one
does not (5 → still every 5).

### Callbacks

| Callback | Purpose |
| --- | --- |
| `BenchmarkImageLogger` | Scores the benchmark sets and writes per-image bicubic\|SR\|HR strips each val run, to every attached TensorBoard or MLflow logger. |
| `SRCheckpoint` | Resumable `.ckpt`. |
| `SRWeightsCheckpoint` | Distributable, optimizer-free `.safetensors` weights, roughly a third the size and safe to hand out without leaking optimizer state. `attribute` picks which component is saved. |
| `SRPredictionWriter` | Writes `sisr predict` output as PNGs. |
| `SRProgressBar` | Progress bar showing the whole run name and only the metric families asked for. |
| `GradNormLogger` | Logs `diag/grad_norm` every `every_n_batches`. |
| `WeightHistogramLogger` | Weight histograms; TensorBoard only. |
| `LossWeightScheduler` | Ramps a named `WeightedSumLoss` term's weight over the run. |
| `MLflowModelRegistrar` | With `register_model: true`, registers the run's final weights as an MLflow model; see [MLflow tracking](#mlflow-tracking). |
| `LearningRateMonitor` | Lightning's, logging per step. |

The callbacks live in the package `sisr.training.callbacks`, one module per concern
(`checkpoint`, `images`, `scalars`, `progress`, `registry`); every public name is importable from
the package, and `class_path: sisr.training.callbacks.<Name>` is the stable path.

Our own cadence arguments are named `every_n_batches` and `every_n_val_runs`, following
Lightning's own `every_n_train_steps` / `every_n_epochs`: the unit is in the name, and none is
called `log_every_n_steps`, which on `Trainer` means the metric *flush* cadence.

Lightning's own names cannot change, so the residual asymmetry stays: `max_steps` and
`every_n_train_steps` count optimizer steps while everything else counts batches. An
adversarial run now **prints the conversion at start** against its own numbers, rather
than relying on you to have read this page.

Set `monitor_metric` for a metric-monitored top-k, or `monitor_metric: null` with
`keep_last` for rolling last-N saves by step. **A monitor's direction is validated at
setup**, so a lower-is-better metric left at the default `mode: max` is refused rather
than quietly keeping the worst model of the run.

Throughput diagnostics are left commented out in the templates because they add
per-step logging overhead. `DeviceStatsMonitor` works as-is; `ThroughputMonitor`
additionally needs a `batch_size_fn`, since batches are `(lr, hr)` tuples rather than
plain tensors, and no project-side helper exists yet.

---

## MLflow tracking

MLflow is an optional second sink beside TensorBoard: `pip install '.[mlflow]'`, then attach
Lightning's own logger. No shipped template enables it, and the SRResNet template carries the
block commented out so tracking is always an explicit choice:

```yaml
trainer:
  logger:
    - class_path: lightning.pytorch.loggers.TensorBoardLogger
      init_args: {save_dir: experiments, name: SRResNet}
    - class_path: lightning.pytorch.loggers.MLFlowLogger
      init_args: {experiment_name: sisr, tracking_uri: http://127.0.0.1:5000}
```

Scalars land in both loggers. The benchmark strips and per-image scalars reach MLflow as run
artifacts and metrics **on the same batch-counted step** TensorBoard gets, so an artifact lines up
against its metric curve; weight histograms stay TensorBoard-only. Without the extra installed,
using MLflow raises an error that names it. A local backend needs a database
(`sqlite:///mlflow.db`): MLflow 3 refuses its plain file store unless `MLFLOW_ALLOW_FILE_STORE=true`.

Registration is off by default. Add `sisr.training.callbacks.MLflowModelRegistrar` with
`register_model: true` and the **final** weights are written at the end of the fit (not a
monitored best checkpoint), logged with their header and registered as
`sisr-<variant_tag><name_suffix>`, tagged `origin=trained`. An already parity-scored external file
registers under `sisr-<variant_tag>-official` with `sisr.training.callbacks.register_official_weights`,
so the two are distinct in the registry.

## Overriding config from the command line

**A dotted override changes only the field it names.** `sisr validate --config my.yaml
--model.init_args.eval_config.ssim_impl=wang` keeps `eval_config` as the subclass the YAML selected
(`SRResNetEvalConfig`), so its other overridden defaults (`crop_border`, `psnr_channels`)
survive; the same holds for `training_config`. The CLI enables subclass mode for
`SREvalConfig` and `SRTrainingConfig` for this reason. Two other forms work and are useful when
changing several fields at once: set them in YAML (the config itself or a second `--config`
overlay), or pass the whole object as one JSON argument with its `class_path`:

```bash
sisr validate --config templates/config.srresnet.template.yaml \
  --model.init_args.eval_config='{"class_path": "sisr.models.srresnet.SRResNetEvalConfig", "init_args": {"ssim_impl": "wang"}}'
```

`--print_config` shows which form survived: the resolved `eval_config` has a `class_path` when the
subclass survived and a bare key/value mapping when it did not.

**`--ckpt_path` preserves the subclass too.** `SRLightning` stores `eval_config` in `hparams` as a
bare dict with no `class_path`; the CLI merges that dict, through the subcommand's own parser, onto
the class `--config` already selected. Stored keys land as `init_args` overrides and every key the
dict omits keeps the subclass's own default, so a checkpoint that predates a field still reloads
with that class's default. This holds for every subcommand that takes `--ckpt_path`, `fit` resume
included.

**Exception under SRGAN.** `SRGANLightning` types `training_config` as
`AdversarialTrainingConfig | None` and `eval_config` as the base `SREvalConfig | None`. The
template's `class_path` pins `training_config` to `SRGANTrainingConfig`, so a dotted override of
`adversarial_weight` or `d_steps_per_g_step` is safe. A dotted override of an `eval_config` field
is not: it rebuilds from the base class, so `perceptual_metrics` empties (silently removing the
metrics that track an adversarial objective) and `ssim_impl` flips from `daala` to `wang`. Use
YAML or the whole-object JSON form for any `eval_config` field there.

## Export

```bash
pip install ".[export]"
sisr export --config my.yaml --ckpt_path best.ckpt --output_path model.onnx
```

The exported graph is the **bare model**, with no processor attached, and it accepts arbitrary
input sizes. For RGB models it is the whole pipeline. For Y-channel models (SRCNN, VDSR, DRCN) the
consumer extracts Y and recombines chroma itself; `sisr/processors/y_channel.py` and
`sisr/processors/studio_y_channel.py` are the reference implementations. `sisr.export.to_onnx()`
does the same from Python.

## SRGAN specifics

`templates/config.srgan.template.yaml` reproduces **SRGAN-VGG54**, the paper's headline variant: the
SRResNet generator, a VGG19 `φ_{5,4}` content loss, and `sisr.losses.AdversarialLoss`
(non-saturating, over the critic's logits) weighted `1e-3`, alternating one `SRDiscriminator`
update per generator update. `sisr.training.SRGANLightning` owns both optimizers under manual
optimization and is selected by `class_path` on the top-level `model:` block. It needs
`pip install '.[perceptual]'` for LPIPS. Its eval config adds `lpips` and `dists` (logged as
`lpips/val` and `dists/val`, and per benchmark set); both are lower-is-better, so a checkpoint
monitoring one needs `mode: min`, which `SRCheckpoint` enforces at setup. Human mean-opinion-score
studies are not reproduced; LPIPS and DISTS correlate with human judgement better than PSNR does,
which is the only claim made for them.

### Two optimizers

The top-level `optimizer:` and `lr_scheduler:` are the **generator's**. The
discriminator's are nested under `model.init_args` as `discriminator_optimizer` and
`discriminator_lr_scheduler`.

That asymmetry is deliberate. A second top-level key would need another
`parser.add_optimizer_args(link_to=...)` in the CLI, and that link targets an argument
`SRLightning` does not have — adding it would break every non-GAN config. Nesting keeps
the second optimizer local to the only module that has one.

**Both schedulers use the same milestones**, so the two networks decay together. Ledig
et al. describe the SRGAN networks as a whole ("1e5 update iterations at 1e-4 and
another 1e5 at 1e-5"); leaving the discriminator's unset would hold the critic at 1e-4
through the whole second phase while the generator it scores drops to 1e-5 — 10× faster,
and a standard way to saturate the generator's adversarial gradient.

`SRGANLightning` steps its schedulers by hand in `on_train_batch_end`, because manual
optimization does not step them. **Milestones are therefore counted in batches.**

### `init_from`

Ledig et al. scope the MSE-initialisation trick to "when training the actual GAN", so a
paper-faithful run starts from an MSE-trained SRResNet. Point `init_from` at a **bare
weights `.safetensors`**, never the sibling `.ckpt` — a `.ckpt` holds the whole LightningModule.

The generator's architecture, processor, output range and scale are all checked against
the file's own metadata and **refused on any mismatch**: weights trained under a
different one produce a model that trains and scores without ever erroring.

Set it to `null` *in the file* (or a `--config` overlay) to train from scratch, which is
not the paper's recipe. `--...init_from=null` on the command line does **not** work:
jsonargparse coerces it to the string `'None'`.

### The discriminator's input size

`SRDiscriminator`'s dense head fixes the accepted input size, so `hr_input_size` must
equal the HR crop the training data serves. The template anchors
`data.train_dataset.init_args.hr_crop_size` to that same value with a YAML anchor, and
the setup probe re-checks it against a real sample before the first step. A mismatch is
otherwise a raw `Linear` shape error.

### One device only

`SRGANLightning` refuses `world_size > 1`. Manual optimization opts out of Lightning's
gradient synchronisation, so the two networks would train out of sync across ranks with
nothing failing.

### Checkpointing an adversarial run

The primary artifacts are **rolling and monitor-free**. PSNR and SSIM get worse by
design under an adversarial objective, so a top-k on either would keep the *least*
adversarial state of the run — typically one from the first few thousand steps.

Weights are saved for **both** networks. The generator's is what gets handed out and
what `init_from` consumes; the discriminator's is kept so the critic can be restored by
hand, though nothing shipped consumes it today. Distinct `filename_prefix` values keep
each callback's rolling deletion off the other's files in a shared directory.

A perceptual monitor does track the objective, but `lpips`/`dists` are lower-is-better,
so `mode` **must** be `min`.

**`GradNormLogger` is deliberately absent from the SRGAN template.** Measured: its
`on_after_backward` hook fires twice per batch, once per network's backward, and
`pl_module.parameters()` spans both networks (58 tensors = 24 generator + 34
discriminator). The single `diag/grad_norm` scalar therefore mixes the two and alternates
between two different meanings. Log per-network norms by hand if you need them.

---

## Project configuration

Notes on `pyproject.toml` and CI that are easy to change without realising what they
were for.

### Dependency floors

- **`torch>=2.6` is a security floor, not a convenience one.** `torch.load(weights_only=True)`
  is this project's load-time safety contract for `.ckpt` files, which stay pickles,
  and GHSA-53q9-r3pm-6pq6 / CVE-2025-32434 lets that check be bypassed for arbitrary code
  execution on torch ≤ 2.5.1. Do not lower it.
- `torchmetrics>=1.9`, `jsonargparse>=4.50` — the versions the relevant APIs were verified
  present in here. Lowering either means testing the lower version, not guessing.

### Extras

Opt-in extras: `export` (onnx, onnxruntime), `perceptual` (LPIPS, pyiqa for Ma and PI, SciPy for
NIQE; DISTS needs no extra), `mlflow` (experiment tracking and the model registry) and `dev` (tests and
lint). The heavy ones are imported only where used.

### Test and lint settings

- `--import-mode=importlib` avoids duplicate-basename collisions across test subdirectories.
  It does **not** add the repo root to `sys.path`, hence the explicit `pythonpath = ["."]` —
  without it, a bare `pytest` from a git worktree would resolve `tests.reference` from a
  different checkout while the sibling expectations stay worktree-local, silently mixing two
  trees.
- `line-length = 100` matches the code as written: 88 would reflow ~157 lines, 100 only ~53.
- Test parallelism is a CI-only concern, and `-n 2` there is deliberate rather than shy —
  every xdist worker is a fresh interpreter that cold-imports the whole torch + lightning
  stack (~10 s) while the suite's own compute is only ~20 s, so past a couple of workers
  the startup costs more than the parallelism wins back.

### CI

Four checks are **required** for a pull request to merge: `test`, `build`, `lint` and
`typecheck`.

- **`typecheck` runs on `windows-latest`, which is not arbitrary.** The cache's liveness
  check uses `ctypes.WinDLL`, genuine platform-specific code guarded at runtime by
  `sys.platform == "win32"` — but typeshed only exposes `WinDLL` under that platform, so
  mypy raises `attr-defined` errors for it anywhere else.
- `actions: write` is granted **per job**, only to the ones that save an Actions cache, so
  the gate jobs cannot write to it at all.
- **Action pinning follows the risk, not a blanket rule.** Third-party actions
  (`dorny/paths-filter`, `codecov/codecov-action`) are pinned to a commit SHA, because a
  tag is mutable and those are the ones worth hardening. First-party `actions/*` stay on
  version tags, and Dependabot keeps them current. `astral-sh/setup-uv` pins an exact
  patch rather than a major tag only because it stopped publishing moving major tags at
  v8.0.0 — not as extra hardening.
- The `changes` job is an **allowlist**, not a denylist: anything not explicitly listed as
  documentation is treated as code and runs the full matrix.
- `codecov/patch` is expected to fail and is not a required check — it reports on paths that
  cannot execute on CPU-only runners. The real coverage gate is `--cov-fail-under=90` inside
  `test`.
