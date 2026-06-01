"""Per-batch colorspace adapters partitioned by model IO colorspace."""
from .base import SRProcessor
from .rgb import RGBProcessor

__all__ = ["SRProcessor", "RGBProcessor"]
