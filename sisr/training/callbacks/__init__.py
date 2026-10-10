"""Lightning callbacks for SR training.

:class:`BenchmarkImageLogger` logs per-image bicubic|SR|HR strips and PSNR/SSIM
for held-out test sets (Set5, Set14, ...) during both ``cli fit`` (every
N val cycles) and ``cli test`` (one-shot final eval).
:class:`GradNormLogger` and :class:`WeightHistogramLogger` log diagnostic
training signals; :class:`LossWeightScheduler` ramps a named term's weight.
:class:`SRCheckpoint` is a thin
:class:`~lightning.pytorch.callbacks.ModelCheckpoint` preset for SR metrics;
:class:`SRWeightsCheckpoint` is a sibling preset that saves bare, optimizer-free
safetensors weights instead; :class:`SRPredictionWriter` writes ``cli predict``
output to disk as PNGs.
"""

from .checkpoint import SRCheckpoint, SRWeightsCheckpoint
from .images import BenchmarkImageLogger, BenchmarkSample, SRPredictionWriter
from .progress import SRProgressBar
from .scalars import GradNormLogger, LossWeightScheduler, WeightHistogramLogger

__all__ = [
    "BenchmarkImageLogger",
    "BenchmarkSample",
    "GradNormLogger",
    "LossWeightScheduler",
    "SRCheckpoint",
    "SRPredictionWriter",
    "SRProgressBar",
    "SRWeightsCheckpoint",
    "WeightHistogramLogger",
]
