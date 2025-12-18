"""
Image Processing Module
Parallel image processing system for Food-101 dataset
"""

from .filters import ImageFilters, ImageProcessingPipeline
from .data_loader import Food101DataLoader, prepare_dataset
from .multiprocessing_pipeline import (
    MultiprocessingPipeline,
    MultiprocessingPoolPipeline,
    batch_worker
)
from .concurrent_pipeline import (
    ConcurrentPipeline,
    ThreadPoolPipeline,
    ProcessPoolPipeline,
    AdaptiveConcurrentPipeline
)

__version__ = "1.0.0"
__author__ = "CST435 Student"

__all__ = [
    'ImageFilters',
    'ImageProcessingPipeline',
    'Food101DataLoader',
    'prepare_dataset',
    'MultiprocessingPipeline',
    'MultiprocessingPoolPipeline',
    'ThreadPoolPipeline',
    'ProcessPoolPipeline',
    'AdaptiveConcurrentPipeline',
]
