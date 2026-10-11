"""Smoke test: every public re-export resolves without circular imports."""

import importlib

import pytest


def test_public_exports():
    from sisr.cli import main  # noqa: F401
    from sisr.colorspace import rgb_to_ycbcr, rgb_to_ycbcr_studio, ycbcr_to_rgb  # noqa: F401
    from sisr.datasets.pre_upsampled import TrainDataset, ValidationDataset  # noqa: F401
    from sisr.models.srcnn import SRCNN, SRCNNEvalConfig, SRCNNTrainingConfig  # noqa: F401
    from sisr.models.srresnet.model import (  # noqa: F401
        SRResidualBlock,
        SRResNet,
        SRUpsampleBlock,
    )
    from sisr.training import (  # noqa: F401
        BenchmarkImageLogger,
        GradNormLogger,
        SRCheckpoint,
        SRDataModule,
        SREvalConfig,
        SRGANLightning,
        SRLightning,
        SRTrainingConfig,
        SRWeightsCheckpoint,
        WeightHistogramLogger,
    )
    from sisr.utils.cache import LMDBCache, LMDBCacheBuildContext  # noqa: F401


def test_old_dataset_module_paths_removed():
    """Guards the clean-break requirement: no shim left at the pre-rename dotted paths."""
    for old in ("sisr.datasets.srcnn", "sisr.datasets.srresnet"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old)
