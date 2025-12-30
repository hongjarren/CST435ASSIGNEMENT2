"""
Multiprocessing-based Parallel Image Processing
Implements image processing using Python's multiprocessing module (Pool variant)
"""

import multiprocessing as mp
from typing import List, Dict, Tuple
import cv2
import numpy as np

from filters import ImageProcessingPipeline


def batch_worker(batch_data: Tuple) -> List[Tuple]:
    """
    Worker function for batch processing in multiprocessing pool.
    
    Args:
        batch_data: Tuple of (image_paths, filters_config)
        
    Returns:
        List of processed image tuples
    """
    image_paths, filters_config = batch_data
    pipeline = ImageProcessingPipeline()
    results = []
    
    for image_path in image_paths:
        try:
            _, processed_image = pipeline.process_image(image_path, filters_config)
            results.append((image_path, processed_image))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return results


class MultiprocessingPoolPipeline(ImageProcessingPipeline):
    """Image processing using multiprocessing.Pool for easier management"""
    
    def __init__(self, num_processes: int = None):
        """
        Initialize pool-based multiprocessing pipeline.
        
        Args:
            num_processes: Number of worker processes (None = CPU count)
        """
        super().__init__()
        self.num_processes = num_processes or mp.cpu_count()
    
    def process_images(self, image_paths: List[str], filters_config: Dict,
                      batch_size: int = None, verbose: bool = True) -> List[Tuple[str, np.ndarray]]:
        """
        Process multiple images using multiprocessing.Pool.
        
        Args:
            image_paths: List of image file paths
            filters_config: Dictionary of filter configurations
            batch_size: Images per batch per process (None = len/num_processes)
            verbose: Print progress information
            
        Returns:
            List of (image_path, processed_image) tuples
        """
        if batch_size is None:
            batch_size = max(1, len(image_paths) // (self.num_processes * 4))
        
        if verbose:
            print(f"Starting multiprocessing.Pool pipeline with {self.num_processes} processes")
            print(f"Processing {len(image_paths)} images with batch size {batch_size}...")
        
        # Create batches
        batches = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batches.append((batch, filters_config))
        
        # Process batches using Pool
        all_results = []
        with mp.Pool(processes=self.num_processes) as pool:
            batch_results = pool.map(batch_worker, batches)
        
        # Flatten results
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        if verbose:
            print(f"Successfully processed {len(all_results)} images using Pool")
        
        return all_results
