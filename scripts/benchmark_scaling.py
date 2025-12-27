#!/usr/bin/env python3
"""
Benchmark Scaling Script
Tests the performance of the image processing pipeline with different numbers of workers.
"""

import sys
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import prepare_dataset
from multiprocessing_pipeline import MultiprocessingPipeline, MultiprocessingPoolPipeline
from concurrent_pipeline import ThreadPoolPipeline, ProcessPoolPipeline

def run_benchmark(num_images=50, output_dir='./output/benchmark'):
    """
    Run scaling benchmark.
    """
    print(f"Running benchmark with {num_images} images")
    
    # Prepare dataset
    image_paths = prepare_dataset(num_images=num_images, output_dir='./data')
    
    # Configuration
    worker_counts = [1, 2, 4, 8]
    filters_config = {
        'grayscale': {'enabled': True},
        'gaussian_blur': {'enabled': True, 'sigma': 1.0},
        'edge_detection': {'enabled': True}
    }
    
    results = []
    
    print(f"\n{'Workers':<10} {'Method':<25} {'Time (s)':<10} {'Speedup':<10}")
    print("-" * 60)
    
    # Baseline (1 worker)
    baseline_time = None
    
    for workers in worker_counts:
        # Test ProcessPoolExecutor (Best for CPU bound)
        pipeline = ProcessPoolPipeline(num_workers=workers)
        
        start_time = time.time()
        pipeline.process_images(image_paths, filters_config, verbose=False)
        duration = time.time() - start_time
        
        if workers == 1:
            baseline_time = duration
            speedup = 1.0
        else:
            speedup = baseline_time / duration
            
        print(f"{workers:<10} {'ProcessPool':<25} {duration:<10.2f} {speedup:<10.2f}x")
        
        results.append({
            'workers': workers,
            'method': 'ProcessPool',
            'time': duration,
            'speedup': speedup
        })
        
    print("-" * 60)
    
    # Visualize
    df = pd.DataFrame(results)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['workers'], df['speedup'], 'o-', label='Measured Speedup')
    plt.plot(df['workers'], df['workers'], 'k--', alpha=0.5, label='Ideal Linear Speedup')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup Factor')
    plt.title(f'Scaling Performance (N={num_images} Images)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/scaling_plot.png")
    print(f"\nPlot saved to {output_dir}/scaling_plot.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-images', type=int, default=50)
    parser.add_argument('--output-dir', default='./output/benchmark')
    args = parser.parse_args()
    
    run_benchmark(args.num_images, args.output_dir)
