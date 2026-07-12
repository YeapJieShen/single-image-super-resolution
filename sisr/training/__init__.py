"""Generic Lightning training plumbing — SRLightning, SRDataModule, callbacks.

Re-exports are flat so experiment YAMLs can use short class paths like
``sisr.training.SRLightning`` instead of ``sisr.training.lightning_module.SRLightning``.
"""

from .callbacks import (
    BenchmarkImageLogger,
    GradNormLogger,
    SRCheckpoint,
    WeightHistogramLogger,
)
from .config import SREvalConfig, SRTrainingConfig
from .datamodule import SRDataModule
from .lightning_module import SRLightning

__all__ = [
    "SRLightning",
    "SRDataModule",
    "SRTrainingConfig",
    "SREvalConfig",
    "BenchmarkImageLogger",
    "GradNormLogger",
    "SRCheckpoint",
    "WeightHistogramLogger",
]
