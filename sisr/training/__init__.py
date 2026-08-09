"""Generic Lightning training plumbing — SRLightning, SRDataModule, callbacks.

Re-exports are flat so experiment YAMLs can use short class paths like
``sisr.training.SRLightning`` instead of ``sisr.training.lightning_module.SRLightning``.
"""

from .callbacks import (
    BenchmarkImageLogger,
    GradNormLogger,
    SRCheckpoint,
    SRPredictionWriter,
    SRWeightsCheckpoint,
    WeightHistogramLogger,
)
from .config import SREvalConfig, SRTrainingConfig
from .datamodule import SRDataModule
from .lightning_module import SRLightning

__all__ = [
    "SRLightning",
    "SRGANLightning",
    "SRDataModule",
    "SRTrainingConfig",
    "SREvalConfig",
    "BenchmarkImageLogger",
    "GradNormLogger",
    "SRCheckpoint",
    "SRWeightsCheckpoint",
    "SRPredictionWriter",
    "WeightHistogramLogger",
]


def __getattr__(name: str):
    """Resolve ``SRGANLightning`` on first access, not at package import.

    Every other re-export above is a plain import; this one cannot be. Each
    per-architecture config (``sisr.models.srgan.config``,
    ``sisr.models.srresnet.config``, ...) imports ``sisr.training.config``,
    which runs *this* module first — so an eager ``from .gan_module import
    SRGANLightning`` here re-enters a half-built ``sisr.models.srgan`` (or
    ``sisr.models.srresnet``) whenever a model package is what got imported
    first, which is what the model-side tests and any ``class_path`` do.
    Deferring to first attribute access resolves the cycle without weakening
    ``SRGANLightning``'s own signature to base types: by the time anything asks
    for the class, both packages are fully imported. ``from sisr.training
    import SRGANLightning`` and ``class_path: sisr.training.SRGANLightning``
    both go through here.

    Args:
        name: Attribute being looked up on this package.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: For any name other than ``SRGANLightning``.
    """
    if name == "SRGANLightning":
        from .gan_module import SRGANLightning

        return SRGANLightning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
