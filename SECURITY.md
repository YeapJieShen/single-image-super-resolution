# Security

## Supported versions

This project is pre-release (`0.1.0`, alpha) and publishes no tagged releases. Only the
current `main` branch is supported. Fixes land on `main`; there are no backports.

## Reporting a vulnerability

Report privately through GitHub's
[security advisory form](https://github.com/YeapJieShen/single-image-super-resolution/security/advisories/new).
Please do not open a public issue for a suspected vulnerability.

## Threat model

The realistic risk in a super-resolution framework is **loading someone else's model
file**. Checkpoints get shared, and a PyTorch checkpoint is a pickle: deserializing an
untrusted one can execute arbitrary code.

### What this project does

Every checkpoint read in `sisr/` passes `weights_only=True`, which restricts
deserialization to plain tensors and primitives instead of arbitrary pickled objects:

- `sisr/export.py` — loading a checkpoint before ONNX export.
- `sisr/cli.py` — reading stored hyperparameters for `--ckpt_path`.

That constraint shapes the code rather than being bolted on. `sisr/training/metadata.py`
writes provenance as **plain types only** — no live objects, no dataclass instances — and
coerces `torch.__version__` with `str()` because `TorchVersion` is a `str` subclass that
`weights_only=True` refuses to unpickle. Configs are stored via `dataclasses.asdict`
rather than as objects for the same reason. Keeping artifacts loadable under
`weights_only=True` is a deliberate contract, so a change that breaks it is a security
regression, not an inconvenience.

`torch>=2.6` is a hard floor in `pyproject.toml`, because earlier versions allowed that
flag to be bypassed (GHSA-53q9-r3pm-6pq6 / CVE-2025-32434).

Provenance metadata deliberately **excludes dataset paths**, so a shared checkpoint does
not leak local filesystem layout.

### What it does not do

- **It does not make third-party checkpoints safe.** `weights_only=True` narrows the
  attack surface; it is not a sandbox. Treat a checkpoint from an untrusted source the
  way you would treat an executable from one.
- **Resuming training through PyTorch Lightning loads a full checkpoint**, including
  optimizer and callback state, and that path is outside this project's control. Only
  resume from checkpoints you produced or trust.
- **`sisr predict` reads arbitrary image files** via Pillow. Image decoders have their own
  history of vulnerabilities; keep Pillow current.
- **Config files are executable by intent.** A YAML passed to `sisr fit` names Python
  classes to import and instantiate via `class_path`. Running an untrusted config is
  equivalent to running untrusted code — by design, since that is how the framework
  selects architectures.
