#!/usr/bin/env python
"""
Quick Start Script - Parallel Image Processing Assignment
Run this to quickly test the system
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*60)
    print("PARALLEL IMAGE PROCESSING - QUICK START")
    print("="*60)
    
    # Check if requirements are installed
    print("\n1. Installing dependencies...")
    try:
        import cv2
        import numpy
        import scipy
        print("   ✓ All dependencies already installed")
    except ImportError:
        print("   Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("   ✓ Dependencies installed")
    
    # Run quick test
    print("\n2. Running quick test (20 images, basic filters)...")
    print("-"*60)
    
    result = subprocess.run([
        sys.executable, "main.py",
        "--num-images", "20",
        "--filter-type", "basic",
        "--test-single",
        "--pipeline", "processpool"
    ], cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("-"*60)
        print("\n✓ Quick test completed successfully!")
        print("\nNext steps:")
        print("  1. Check output directory for processed images")
        print("  2. Run full comparison: python main.py --num-images 50")
        print("  3. Check README.md for advanced usage")
    else:
        print("\n✗ Test failed. Check the error output above.")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
