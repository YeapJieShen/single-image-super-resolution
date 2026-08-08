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
- **`sisr` console script** over a single `LightningCLI` entrypoint, with `fit`, `test`,
  `predict` and `export` subcommands. One self-contained YAML drives an experiment.
- **Pluggable losses** (`sisr/losses/`): `CharbonnierLoss`, `TotalVariationLoss`,
  `VGG19FeatureLoss`, `VGG16FeatureLoss` and `WeightedSumLoss`, selectable and combinable
  from YAML. L1 and MSE need no wrapper — `torch.nn.L1Loss` already works as a criterion.
- **daala-methodology SSIM** (`sisr/ssim.py`) behind `SREvalConfig.ssim_impl`, the
  convention Ledig et al. used, verified against daala's own C implementation.
  `SRResNetEvalConfig` defaults to it; everything else stays on the field-standard fixed
  11×11 gaussian.
- **Vendored byte-exact MATLAB `imresize`** (`sisr/imresize.py`, MIT) as the sole LR
  degradation, verified byte-identical to real MATLAB output across the benchmark sets.
- **ONNX export** (`sisr/export.py`) behind the `[export]` extra, with dynamic spatial
  axes and provenance written into `metadata_props`.
- **Provenance metadata** (`sisr/training/metadata.py`) — one builder feeding checkpoints,
  bare weight files, and ONNX exports, so the three cannot drift.
- **`SRWeightsCheckpoint`** — distributable optimizer-free `.pt` weights, roughly a third
  the size of a full checkpoint.
- **Opt-in full-step CUDA-graph capture** (`SRTrainingConfig.cuda_graph`), which refuses
  configurations it cannot capture soundly rather than capturing them anyway.
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

### Fixed

- **Graphed training froze at the first mid-training validation.** Lightning's pre-val
  `zero_grad(set_to_none=True)` severed a live CUDA graph's gradient tensors from the
  optimizer, so weights stopped updating while the reported loss kept moving.
- **`--ckpt_path` could not reload any checkpoint this project had ever saved.**
- **The LMDB cache could delete another process's in-progress build.** Deletion is now
  restricted to proven corruption; environmental failures raise instead.
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
