"""
concurrent.futures-based Parallel Image Processing
Implements image processing using Python's concurrent.futures module
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Callable
import time
import os
import cv2
import numpy as np
from pathlib import Path

from filters import ImageProcessingPipeline


class ConcurrentPipeline(ImageProcessingPipeline):
    """Base class for concurrent.futures-based pipelines"""
    
    def __init__(self, num_workers: int = None):
        """
        Initialize concurrent pipeline.
        
        Args:
            num_workers: Number of worker threads/processes (None = optimal)
        """
        super().__init__()
        self.num_workers = num_workers
    
    def _process_single_image(self, image_path: str, filters_config: Dict) -> Tuple:
        """
        Process a single image (can be called by executor).
        
        Args:
            image_path: Path to the image
            filters_config: Filter configuration
            
        Returns:
            Tuple of (image_path, processed_image, error)
        """
        try:
            _, processed_image = self.process_image(image_path, filters_config)
            return (image_path, processed_image, None)
        except Exception as e:
            return (image_path, None, str(e))


class ThreadPoolPipeline(ConcurrentPipeline):
    """Image processing using ThreadPoolExecutor"""
    
    def process_images(self, image_paths: List[str], filters_config: Dict,
                      verbose: bool = True) -> List[Tuple[str, np.ndarray]]:
        """
        Process multiple images in parallel using threads.
        
        Note: Due to Python's GIL, this is better for I/O-bound operations
        but for CPU-intensive image processing, ProcessPoolExecutor is recommended.
        
        Args:
            image_paths: List of image file paths
            filters_config: Dictionary of filter configurations
            verbose: Print progress information
            
        Returns:
            List of (image_path, processed_image) tuples
        """
        results = []
        errors = []
        
        if verbose:
            print(f"Starting ThreadPoolExecutor with {self.num_workers or 'optimal'} threads")
            print(f"Processing {len(image_paths)} images...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_single_image, path, filters_config): path
                for path in image_paths
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                completed += 1
                image_path, processed_image, error = future.result()
                
                if error is None:
                    results.append((image_path, processed_image))
                else:
                    errors.append((image_path, error))
                    if verbose:
                        print(f"Error processing {image_path}: {error}")
                
                if verbose and completed % 10 == 0:
                    print(f"  Completed {completed}/{len(image_paths)} images")
        
        if verbose:
            print(f"Successfully processed {len(results)} images using ThreadPoolExecutor")
            if errors:
                print(f"Failed to process {len(errors)} images")
        
        return results


class ProcessPoolPipeline(ConcurrentPipeline):
    """Image processing using ProcessPoolExecutor (true parallelism)"""
    
    def process_images(self, image_paths: List[str], filters_config: Dict,
                      verbose: bool = True) -> List[Tuple[str, np.ndarray]]:
        """
        Process multiple images in parallel using processes.
        
        This provides true parallelism and is recommended for CPU-intensive
        image processing operations.
        
        Args:
            image_paths: List of image file paths
            filters_config: Dictionary of filter configurations
            verbose: Print progress information
            
        Returns:
            List of (image_path, processed_image) tuples
        """
        results = []
        errors = []
        
        # Determine number of workers
        max_workers = self.num_workers or os.cpu_count()
        
        if verbose:
            print(f"Starting ProcessPoolExecutor with {max_workers} processes")
            print(f"Processing {len(image_paths)} images...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_single_image, path, filters_config): path
                for path in image_paths
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    image_path, processed_image, error = future.result()
                    
                    if error is None:
                        results.append((image_path, processed_image))
                    else:
                        errors.append((image_path, error))
                        if verbose:
                            print(f"Error processing {image_path}: {error}")
                except Exception as e:
                    if verbose:
                        print(f"Exception in future: {e}")
                
                if verbose and completed % 10 == 0:
                    print(f"  Completed {completed}/{len(image_paths)} images")
        
        if verbose:
            print(f"Successfully processed {len(results)} images using ProcessPoolExecutor")
            if errors:
                print(f"Failed to process {len(errors)} images")
        
        return results


class AdaptiveConcurrentPipeline(ConcurrentPipeline):
    """
    Adaptive pipeline that chooses between ThreadPoolExecutor and ProcessPoolExecutor
    based on the number of images and available resources.
    """
    
    def process_images(self, image_paths: List[str], filters_config: Dict,
                      verbose: bool = True) -> List[Tuple[str, np.ndarray]]:
        """
        Process images using adaptive strategy.
        
        Uses ThreadPoolExecutor for small batches (< 50 images) and 
        ProcessPoolExecutor for larger batches.
        
        Args:
            image_paths: List of image file paths
            filters_config: Dictionary of filter configurations
            verbose: Print progress information
            
        Returns:
            List of (image_path, processed_image) tuples
        """
        # Choose executor based on workload
        use_processes = len(image_paths) > 50
        
        if use_processes:
            if verbose:
                print("Using ProcessPoolExecutor for CPU-intensive parallel processing")
            pipeline = ProcessPoolPipeline(num_workers=self.num_workers)
        else:
            if verbose:
                print("Using ThreadPoolExecutor for smaller workload")
            pipeline = ThreadPoolPipeline(num_workers=self.num_workers)
        
        return pipeline.process_images(image_paths, filters_config, verbose=verbose)
