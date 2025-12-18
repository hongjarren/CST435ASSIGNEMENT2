"""
Image Filtering Operations Module
Implements various image filters for parallel processing
"""

import numpy as np
from scipy import ndimage, signal
from PIL import Image
from typing import Tuple
import cv2


class ImageFilters:
    """Collection of image filtering operations"""
    
    @staticmethod
    def grayscale_conversion(image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to grayscale using luminance formula.
        
        Formula: Gray = 0.299 * R + 0.587 * G + 0.114 * B
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Grayscale image array (H, W)
        """
        if len(image.shape) == 2:
            return image  # Already grayscale
        
        # Luminance formula for better perceptual grayscale conversion
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return gray.astype(np.uint8)
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur using the exact 3×3 Gaussian kernel.

        Kernel enforced (normalized):
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]] / 16

        Notes:
            - The 'kernel_size' and 'sigma' parameters are ignored to strictly
              comply with the assignment's 3×3 kernel requirement.
            - Works for both grayscale (H, W) and RGB (H, W, 3) images.

        Args:
            image: Input image array (uint8 recommended)
            kernel_size: Ignored (kept for API compatibility)
            sigma: Ignored (kept for API compatibility)

        Returns:
            Blurred image array (uint8)
        """
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 16.0

        # Use OpenCV's filter2D to apply the exact kernel; keep original depth
        blurred = cv2.filter2D(src=image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
        return blurred.astype(np.uint8)
    
    @staticmethod
    def edge_detection_sobel(image: np.ndarray) -> np.ndarray:
        """
        Apply Sobel filter to detect edges.
        
        Computes gradients in X and Y directions using Sobel operators.
        
        Args:
            image: Input image array (preferably grayscale)
            
        Returns:
            Edge-detected image array
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = ImageFilters.grayscale_conversion(image)
        
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
        
        # Apply Sobel filters
        edges_x = signal.convolve2d(image.astype(float), sobel_x, mode='same')
        edges_y = signal.convolve2d(image.astype(float), sobel_y, mode='same')
        
        # Compute edge magnitude
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        # Normalize to 0-255 range
        edges = (edges / edges.max() * 255).astype(np.uint8)
        return edges
    
    @staticmethod
    def image_sharpening(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Enhance edges and details through sharpening.
        
        Uses unsharp masking technique.
        
        Args:
            image: Input image array
            strength: Sharpening strength (0.5-2.0 recommended)
            
        Returns:
            Sharpened image array
        """
        # Create blurred version
        blurred = ndimage.gaussian_filter(image.astype(float), sigma=1.0)
        
        # Unsharp mask: Original + (Original - Blurred) * strength
        sharpened = image.astype(float) + (image.astype(float) - blurred) * strength
        
        # Clip values to valid range
        sharpened = np.clip(sharpened, 0, 255)
        return sharpened.astype(np.uint8)
    
    @staticmethod
    def brightness_adjustment(image: np.ndarray, delta: int = 0) -> np.ndarray:
        """
        Increase or decrease image brightness.
        
        Args:
            image: Input image array
            delta: Brightness adjustment value (-127 to 127)
            
        Returns:
            Brightness-adjusted image array
        """
        # Clip delta to valid range
        delta = np.clip(delta, -127, 127)
        
        # Apply brightness adjustment
        adjusted = image.astype(float) + delta
        
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255)
        return adjusted.astype(np.uint8)


class ImageProcessingPipeline:
    """Base class for image processing pipelines"""
    
    def __init__(self):
        self.filters = ImageFilters()
    
    def process_image(self, image_path: str, filters_config: dict) -> Tuple[str, np.ndarray]:
        """
        Apply a sequence of filters to an image.
        
        Args:
            image_path: Path to the input image
            filters_config: Dictionary specifying which filters to apply and their parameters
            
        Returns:
            Tuple of (original_image_path, processed_image)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply filters in sequence
        result = image.copy()
        
        for filter_name, params in filters_config.items():
            if filter_name == 'grayscale' and params.get('enabled', False):
                result = self.filters.grayscale_conversion(result)
                
            elif filter_name == 'gaussian_blur' and params.get('enabled', False):
                sigma = params.get('sigma', 1.0)
                result = self.filters.gaussian_blur(result, sigma=sigma)
                
            elif filter_name == 'edge_detection' and params.get('enabled', False):
                result = self.filters.edge_detection_sobel(result)
                
            elif filter_name == 'sharpening' and params.get('enabled', False):
                strength = params.get('strength', 1.0)
                result = self.filters.image_sharpening(result, strength=strength)
                
            elif filter_name == 'brightness' and params.get('enabled', False):
                delta = params.get('delta', 0)
                result = self.filters.brightness_adjustment(result, delta=delta)
        
        return image_path, result
