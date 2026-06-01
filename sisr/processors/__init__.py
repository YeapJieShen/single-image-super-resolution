"""Per-batch colorspace adapters partitioned by model IO colorspace."""
from .base import SRProcessor
from .rgb import RGBProcessor
from .y_channel import YChannelProcessor
from .ycbcr import YCbCrProcessor

__all__ = ["SRProcessor", "RGBProcessor", "YChannelProcessor", "YCbCrProcessor"]
