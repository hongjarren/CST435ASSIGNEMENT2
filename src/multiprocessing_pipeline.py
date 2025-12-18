"""
Multiprocessing-based Parallel Image Processing
Implements image processing using Python's multiprocessing module
"""

import multiprocessing as mp
import os
from typing import List, Dict, Tuple
import time
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from filters import ImageProcessingPipeline


class MultiprocessingPipeline(ImageProcessingPipeline):
    """Image processing pipeline using multiprocessing"""
    
    def __init__(self, num_processes: int = None):
        """
        Initialize multiprocessing pipeline.
        
        Args:
            num_processes: Number of worker processes (None = CPU count)
        """
        super().__init__()
        self.num_processes = num_processes or mp.cpu_count()
    
    def process_images(self, image_paths: List[str], filters_config: Dict, 
                      verbose: bool = True) -> List[Tuple[str, np.ndarray]]:
        """
        Process multiple images in parallel using multiprocessing.
        Uses ProcessPoolExecutor which handles Windows better than direct Process management.
        
        Args:
            image_paths: List of image file paths
            filters_config: Dictionary of filter configurations
            verbose: Print progress information
            
        Returns:
            List of (image_path, processed_image) tuples
        """
        if verbose:
            print(f"Starting multiprocessing pipeline with {self.num_processes} processes")
            print(f"Processing {len(image_paths)} images...")
        
        results = []
        errors = []
        
        # Use ProcessPoolExecutor for better Windows compatibility
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._worker_wrapper, path, filters_config): (idx, path)
                for idx, path in enumerate(image_paths)
            }
            
            # Collect results as they complete
            completed = 0
            for future in futures:
                completed += 1
                idx, image_path = futures[future]
                
                try:
                    processed_image, error = future.result(timeout=300)  # 5 minute timeout
                    
                    if error is None:
                        results.append((image_path, processed_image))
                    else:
                        errors.append((image_path, error))
                        if verbose:
                            print(f"Error processing {image_path}: {error}")
                except Exception as e:
                    errors.append((image_path, str(e)))
                    if verbose:
                        print(f"Exception processing {image_path}: {e}")
                
                if verbose and completed % 10 == 0:
                    print(f"  Completed {completed}/{len(image_paths)} images")
        
        if verbose:
            print(f"Successfully processed {len(results)} images")
            if errors:
                print(f"Failed to process {len(errors)} images")
        
        return results
    
    @staticmethod
    def _worker_wrapper(image_path: str, filters_config: dict) -> Tuple:
        """
        Static wrapper for worker process (for pickling compatibility).
        
        Args:
            image_path: Path to image
            filters_config: Filter configuration
            
        Returns:
            Tuple of (processed_image, error)
        """
        try:
            pipeline = ImageProcessingPipeline()
            _, processed_image = pipeline.process_image(image_path, filters_config)
            return processed_image, None
        except Exception as e:
            return None, str(e)


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
