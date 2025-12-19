"""
Data Loader for Food-101 Dataset
Handles downloading and loading images from the Food-101 dataset
"""

import os
from pathlib import Path
from typing import List, Tuple
import urllib.request
import zipfile
import json
import random
from PIL import Image
import cv2


class Food101DataLoader:
    """Data loader for Food-101 dataset"""
    
    # Food-101 dataset information
    DATASET_URL = "https://www.kaggle.com/datasets/dansbecker/food-101"
    DATASET_DIRNAME = "food-101"
    
    def __init__(self, data_path: str = "./data"):
        """
        Initialize Food-101 data loader.
        
        Args:
            data_path: Path to store dataset
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.data_path / self.DATASET_DIRNAME
    
    def get_sample_images(self, num_images: int = 100, categories: List[str] = None) -> List[str]:
        """
        Get sample images. Prefers real Food-101 dataset if available, otherwise
        generates synthetic images for testing.
        
        Args:
            num_images: Number of sample images to use
            categories: Specific food categories (None = random)
            
        Returns:
            List of image file paths
        """
        # Preferred sources in order: subset_2000, full food-101/images
        preferred_sources = [
            self.data_path / "subset_2000",
            self.dataset_path / "images",
        ]

        for images_root in preferred_sources:
            if images_root.exists() and images_root.is_dir():
                real_images = self._get_food101_real_images(num_images=num_images, categories=categories, root=images_root)
                if real_images:
                    print(f"Found dataset at {images_root}. Using {len(real_images)} images.")
                    return real_images

        # Fall back to synthetic images
        sample_dir = self.data_path / "sample_images"
        sample_dir.mkdir(exist_ok=True)
        existing_images = list(sample_dir.glob("*.jpg"))
        if len(existing_images) >= num_images:
            return [str(img) for img in existing_images[:num_images]]

        print(f"Creating {num_images} synthetic sample images for testing...")
        image_paths: List[str] = []
        for i in range(num_images):
            height, width = 512, 512
            image = self._generate_sample_image(height, width, i)
            image_path = sample_dir / f"sample_{i:04d}.jpg"
            cv2.imwrite(str(image_path), image)
            image_paths.append(str(image_path))
            if (i + 1) % 20 == 0:
                print(f"  Created {i + 1} sample images")

        print(f"Sample images ready at {sample_dir}")
        return image_paths

    def _get_food101_real_images(self, num_images: int, categories: List[str] | None, root: Path | None = None) -> List[str]:
        """
        Return up to num_images image paths from real Food-101 dataset if present.

        Tries to balance selection across categories when possible.
        """
        images_root = root if root is not None else (self.dataset_path / "images")
        if not images_root.exists():
            return []

        # Determine categories
        if categories:
            class_dirs = [images_root / c for c in categories if (images_root / c).is_dir()]
        else:
            class_dirs = [d for d in images_root.iterdir() if d.is_dir()]
            random.shuffle(class_dirs)

        if not class_dirs:
            return []

        # Balanced sampling across classes
        per_class = max(1, (num_images + len(class_dirs) - 1) // len(class_dirs))
        selected: List[str] = []

        for cls in class_dirs:
            imgs = list(cls.glob("*.jpg")) + list(cls.glob("*.jpeg")) + list(cls.glob("*.png"))
            if not imgs:
                continue
            random.shuffle(imgs)
            take = min(per_class, len(imgs))
            selected.extend(str(p) for p in imgs[:take])
            if len(selected) >= num_images:
                break

        # If still short, fill from any remaining images
        if len(selected) < num_images:
            all_imgs = list(images_root.rglob("*.jpg")) + list(images_root.rglob("*.jpeg")) + list(images_root.rglob("*.png"))
            random.shuffle(all_imgs)
            for p in all_imgs:
                sp = str(p)
                if sp not in selected:
                    selected.append(sp)
                if len(selected) >= num_images:
                    break

        return selected[:num_images]
    
    @staticmethod
    def _generate_sample_image(height: int, width: int, seed: int) -> object:
        """
        Generate a synthetic sample image with food-like colors.
        
        Args:
            height: Image height in pixels
            width: Image width in pixels
            seed: Random seed for reproducibility
            
        Returns:
            Image array (BGR format for cv2)
        """
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Create base image with warm food-like colors
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some random colored regions (simulating food items)
        num_regions = random.randint(3, 8)
        food_colors = [
            (0, 100, 200),    # Orange (BGR)
            (50, 100, 200),   # Red-orange
            (100, 150, 100),  # Green
            (200, 100, 50),   # Blue
            (150, 100, 50),   # Purple
        ]
        
        for _ in range(num_regions):
            # Random circular region
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius = random.randint(50, 150)
            color = random.choice(food_colors)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            image[mask] = color
        
        # Add some noise for texture
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add some gaussian blur for smoothness
        from scipy import ndimage
        image = ndimage.gaussian_filter(image, sigma=2)
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def create_test_subset(self, num_images: int = 50) -> List[str]:
        """
        Create a small test subset of images.
        
        Args:
            num_images: Number of images in the test subset
            
        Returns:
            List of image file paths
        """
        return self.get_sample_images(num_images)
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate that an image file is readable.
        
        Args:
            image_path: Path to the image
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            img = cv2.imread(image_path)
            return img is not None and img.size > 0
        except Exception:
            return False
    
    def get_image_info(self, image_path: str) -> dict:
        """
        Get information about an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with image information
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            height, width = img.shape[:2]
            return {
                'path': image_path,
                'width': width,
                'height': height,
                'channels': img.shape[2] if len(img.shape) > 2 else 1,
                'size_bytes': os.path.getsize(image_path)
            }
        except Exception as e:
            return None


def prepare_dataset(num_images: int = 100, output_dir: str = "./data") -> List[str]:
    """
    Prepare dataset for image processing.
    
    Args:
        num_images: Number of images to prepare
        output_dir: Directory to store images
        
    Returns:
        List of image file paths
    """
    loader = Food101DataLoader(output_dir)
    return loader.get_sample_images(num_images)
