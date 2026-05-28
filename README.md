# single-image-super-resolution

A PyTorch Lightning framework for reproducing single image super-resolution (SISR) papers. Training, validation, and test evaluation are all driven through a single `LightningCLI` entrypoint — switching architectures is just a matter of pointing the config at a different model and dataset.

## Models

- **SRCNN** — the 3-layer CNN from [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092). The LR input is pre-upsampled to the HR size (blur → downsample → bicubic upsample), so the model learns a same-resolution refinement. Trained on the Y channel by default.
- **SRResNet** — the residual generator from [Photo-Realistic Single Image Super-Resolution Using a GAN](https://arxiv.org/pdf/1609.04802). The LR input is genuinely small and the model upsamples it ×`scale` via sub-pixel convolution. Trained on RGB.

The generic [`SRLightning`](sisr/training/lightning_module.py) module wraps any SR model, and [`SRDataModule`](sisr/training/datamodule.py) feeds it; per-architecture knobs live in config dataclasses ([`sisr/training/config.py`](sisr/training/config.py) and the SRCNN subclasses in [`sisr/models/srcnn/config.py`](sisr/models/srcnn/config.py)).

## Install

Requires Python ≥ 3.12.

```bash
# CPU / default install (resolves torch from PyPI):
pip install .

# GPU install — same deps, but torch / torchvision come from the PyTorch CUDA index:
pip install -r requirements.txt

# Editable + tests:
pip install -e ".[dev]"
```

All version constraints live in `pyproject.toml`. `requirements.txt` is intentionally minimal — it just adds `--extra-index-url https://download.pytorch.org/whl/cu130` (the CUDA wheels aren't on PyPI) and points pip at the project so the same abstract deps install against the GPU wheel index instead.

## Usage

Training is configured entirely through YAML. Copy a template from [`templates/`](templates/), fill in the `TODO` placeholders (dataset directories, experiment output paths), then:

```bash
# Train SRCNN (x3):
python -m sisr.cli fit --config templates/config.srcnn.template.yaml

# Train SRResNet (x4):
python -m sisr.cli fit --config templates/config.srresnet.template.yaml
```

Per-run overrides can be passed on the CLI without editing the YAML:

```bash
python -m sisr.cli fit --config templates/config.srcnn.template.yaml \
    --trainer.max_steps=500000 --optimizer.init_args.lr=1e-3
```

After training, evaluate a checkpoint against the test sets (Set5, Set14, …):

```bash
python -m sisr.cli test --config templates/config.srcnn.template.yaml \
    --ckpt_path path/to/best.ckpt
```

Test sets are also surfaced during `fit` (monitored each validation cycle), and PSNR/SSIM plus LR│SR│HR image strips are logged to TensorBoard.

## Tests

```bash
pytest
```
