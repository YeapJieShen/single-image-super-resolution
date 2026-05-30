"""Smoke test: every public re-export resolves without circular imports."""


def test_public_exports():
    from sisr.cli import main  # noqa: F401
    from sisr.training import (  # noqa: F401
        BenchmarkImageLogger,
        GradNormLogger,
        SRCheckpoint,
        SRDataModule,
        SREvalConfig,
        SRLightning,
        SRTrainingConfig,
        WeightHistogramLogger,
    )
    from sisr.datasets.srcnn import TrainDataset, ValidationDataset  # noqa: F401
    from sisr.models.srcnn import SRCNN, SRCNNEvalConfig, SRCNNTrainingConfig  # noqa: F401
    from sisr.models.srresnet.model import (  # noqa: F401
        SRResNet,
        SRResidualBlock,
        SRUpsampleBlock,
    )
    from sisr.utils import (  # noqa: F401
        LMDBCache,
        LMDBCacheBuildContext,
        extract_model_input,
        rgb_to_ycbcr,
        reconstruct_sr_rgb,
        ycbcr_to_rgb,
    )
