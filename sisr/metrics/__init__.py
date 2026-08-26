"""Image-quality metrics and the scorer that composes them.

:mod:`~sisr.metrics.ssim` and :mod:`~sisr.metrics.perceptual` are the
primitives; :mod:`~sisr.metrics.scoring` is the orchestrator that scores an
aligned ``(sr_rgb, hr_rgb)`` pair under one evaluation config.

Nothing is re-exported here — import the submodule you want. That matches the
rest of the package, where leaf modules are addressed by full path, and it
keeps this ``__init__`` from importing ``scoring``, whose type-only reference
to ``SREvalConfig`` would otherwise become a runtime import cycle through
:mod:`sisr.training`.
"""
