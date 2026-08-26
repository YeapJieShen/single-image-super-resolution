# Changelog

Notable changes to this project, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning will follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once a
release is tagged. Nothing is released yet: the project is pre-1.0 and every entry below
sits under **Unreleased**. Until a version is cut, any part of the API may change without
notice.

## [Unreleased]

### Added

- **SRResNet** and **SRCNN**, both fully wired — model, dataset pipeline, config
  dataclasses, and an experiment template each.
- **SRGAN** (`sisr/models/srgan/`, `sisr/training/gan_module.py`) — the SRResNet
  generator trained adversarially against an `SRDiscriminator` critic. `SRGANLightning`
  drives both optimizers under manual optimization, `AdversarialLoss` supplies the
  non-saturating objective over the critic's logits, and the shipped template reproduces
  SRGAN-VGG54. The generator initialises from an MSE-trained SRResNet's bare weights, the
  paper's recipe, and refuses any weights file whose architecture, processor, output
  range or scale disagrees with the run loading it. `trainer.global_step` counts
  optimizer steps, so it advances twice per batch here and nowhere else — see the README.
- **Subclass-mode CLI** — the top-level `model:` block may now name its Lightning module
  with `class_path`/`init_args`, which is how a config selects `SRGANLightning` instead
  of `SRLightning`. Existing configs that set `model:` fields directly keep resolving
  against the default `SRLightning`; the shipped templates were re-nested to match the
  new form.
- **Perceptual metrics** (`sisr/metrics/perceptual.py`) — LPIPS and DISTS, selected by
  `SREvalConfig.perceptual_metrics` and logged as `lpips/val` / `dists/val` and per
  benchmark set. Empty by default, so every existing architecture logs exactly the tags
  it logged before. They exist because an adversarial objective makes PSNR and SSIM worse
  by design; neither substitutes for the paper's MOS study. `lpips_net` selects the LPIPS
  backbone, and a figure is comparable only to one computed under the same backbone.
- **`[perceptual]` extra** — `pip install '.[perceptual]'` for LPIPS, which torchmetrics
  gates on the `lpips` package. DISTS needs nothing beyond torchvision and ships in core.
- **Rolling last-N checkpoints** — `SRCheckpoint` and `SRWeightsCheckpoint` with
  `monitor_metric: null` keep the last `keep_last` saves by step, which Lightning cannot
  express through `save_top_k` alone. A metric-monitored "best" is the wrong artifact for
  an adversarial run. When a monitor *is* set, its direction is now validated too, so a
  lower-is-better metric left at the default `mode='max'` is refused at setup instead of
  keeping the worst model of the run.
- **`sisr` console script** over a single `LightningCLI` entrypoint, with `fit`, `test`,
  `predict` and `export` subcommands. One self-contained YAML drives an experiment.
- **Pluggable losses** (`sisr/losses/`): `CharbonnierLoss`, `TotalVariationLoss`,
  `VGG19FeatureLoss`, `VGG16FeatureLoss` and `WeightedSumLoss`, selectable and combinable
  from YAML. L1 and MSE need no wrapper — `torch.nn.L1Loss` already works as a criterion.
- **daala-methodology SSIM** (`sisr/metrics/ssim.py`) behind `SREvalConfig.ssim_impl`, the
  convention Ledig et al. used, verified against daala's own C implementation.
  `SRResNetEvalConfig` defaults to it; everything else stays on the field-standard fixed
  11×11 gaussian.
- **Vendored byte-exact MATLAB `imresize`** (`sisr/utils/imresize.py`, MIT) as the sole LR
  degradation, verified byte-identical to real MATLAB output across the benchmark sets.
- **ONNX export** (`sisr/export.py`) behind the `[export]` extra, with dynamic spatial
  axes and provenance written into `metadata_props`.
- **Provenance metadata** (`sisr/training/metadata.py`) — one builder feeding checkpoints,
  bare weight files, and ONNX exports, so the three cannot drift.
- **`SRWeightsCheckpoint`** — distributable optimizer-free `.pt` weights, roughly a third
  the size of a full checkpoint. `attribute` picks which component gets saved, with
  matching provenance, so a GAN run can hand out the generator and retain the critic in
  separate files.
- **LR-only prediction path** — `PredictDataset`, `predict_step`, and `SRPredictionWriter`.
- **LMDB HR caching** shared by both architectures, with an advisory build lock so
  concurrent builds do not duplicate work.

### Changed

- **Y-channel metrics are computed in BT.601 studio range**, matching MATLAB and the
  published SR literature. Figures produced before this read systematically low.
- **SR output is clamped to `[0, 1]` before scoring.**
- **Both train datasets share one architecture-neutral raw-HR cache.** Cache validity
  depends on the file manifest and a format tag only, so changing scale, crop size or
  stride needs no rebuild, and caching a dataset for one architecture serves the other.
- **Colorspace is chosen by composing an `SRProcessor`**, replacing a config string field.
- **Google-style docstrings are enforced** via ruff's pydocstyle rules.
- **`torchmetrics>=1.9` floor** (was `>=1.4`) — the version DISTS was verified present in
  on this project. Lowering it means testing the lower version, not guessing.

### Fixed

- **`--ckpt_path` could not reload any checkpoint this project had ever saved.**
- **`--ckpt_path` rebuilt an architecture's `eval_config` as the base class**, so every
  default the subclass had overridden reverted on reload — a resumed SRResNet run could
  monitor a different SSIM convention than a fresh run from the same YAML. The
  checkpoint's stored fields are now merged onto the class the config already selected,
  so subclass identity and subclass-only defaults both survive, including from
  checkpoints saved before a field existed. A dotted *command-line* override of such a
  field still reverts the object to its base class — a separate, open defect; see the
  README.
- **The LMDB cache could delete another process's in-progress build.** Deletion is now
  restricted to proven corruption; environmental failures raise instead.
- **A cache build could wait on its lock forever.** The timeout only bounded waits on a
  holder confirmed alive. A sentinel whose pid was unreadable, or was this process's own
  after pid recycling, took neither that path nor the abandoned-lock takeover path while
  a heartbeat kept it looking fresh — so the waiter polled with no error, no progress and
  no further log line. Every wait path is now bounded by the same cap.
- **SRCNN's paper weight-init std** corrected to the published `0.001`.
- **`example_input_shape` pointed at the wrong size**, defeating compile warm-up and
  misreporting FLOPs.

### Removed

- **`opencv-python-headless`** and the `cv2` degradation backend, along with
  `resize_backend` and `blur_sigma`. MATLAB `imresize` is the only degradation path.
- **`albumentationsx`** — AGPL-3.0 is incompatible with this project's MIT licence, and
  its entire use was six primitives with no augmentation.

### Security

- **`torch>=2.6` floor**, excluding the `weights_only=True` bypass
  (GHSA-53q9-r3pm-6pq6 / CVE-2025-32434). See [SECURITY.md](SECURITY.md).

[Unreleased]: https://github.com/YeapJieShen/single-image-super-resolution/commits/main
