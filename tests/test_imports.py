"""Smoke test: every public re-export resolves without circular imports."""


def test_public_exports():
    from sisr.cache import LMDBCache, LMDBCacheBuildContext  # noqa: F401
    from sisr.cli import main  # noqa: F401
    from sisr.colorspace import rgb_to_ycbcr, rgb_to_ycbcr_studio, ycbcr_to_rgb  # noqa: F401
    from sisr.datasets.srcnn import TrainDataset, ValidationDataset  # noqa: F401
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
