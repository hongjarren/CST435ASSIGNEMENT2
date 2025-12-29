"""
Core Scaling Benchmark Script
Runs image processing with different core counts and plots performance
"""

import sys
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import prepare_dataset
from multiprocessing_pipeline import MultiprocessingPoolPipeline
from concurrent_pipeline import ProcessPoolPipeline, ThreadPoolPipeline


def run_benchmark(image_paths, filters_config, core_counts, pipeline_type='processpool'):
    """
    Run benchmark with different core counts.
    
    Args:
        image_paths: List of image paths
        filters_config: Filter configuration
        core_counts: List of core counts to test
        pipeline_type: 'processpool', 'multiprocessing_pool', or 'threadpool'
        
    Returns:
        Dictionary with results
    """
    results = {
        'core_counts': core_counts,
        'times': [],
        'speedups': [],
        'efficiencies': []
    }
    
    print(f"\n{'='*60}")
    print(f"CORE SCALING BENCHMARK - {pipeline_type.upper()}")
    print(f"{'='*60}")
    print(f"Images: {len(image_paths)}")
    print(f"Testing cores: {core_counts}")
    print(f"{'='*60}\n")
    
    # Run sequential baseline (1 core equivalent)
    print("Running sequential baseline...")
    from filters import ImageProcessingPipeline
    baseline_pipeline = ImageProcessingPipeline()
    
    start = time.time()
    for idx, img_path in enumerate(image_paths):
        try:
            baseline_pipeline.process_image(img_path, filters_config)
        except Exception as e:
            print(f"Error: {e}")
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)}")
    
    baseline_time = time.time() - start
    print(f"Sequential baseline: {baseline_time:.2f}s\n")
    
    # Test each core count
    for cores in core_counts:
        print(f"Testing with {cores} cores...")
        
        # Select pipeline
        if pipeline_type == 'processpool':
            pipeline = ProcessPoolPipeline(num_workers=cores)
        elif pipeline_type == 'multiprocessing_pool':
            pipeline = MultiprocessingPoolPipeline(num_processes=cores)
        elif pipeline_type == 'threadpool':
            pipeline = ThreadPoolPipeline(num_workers=cores)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_type}")
        
        start = time.time()
        pipeline.process_images(image_paths, filters_config, verbose=False)
        elapsed = time.time() - start
        
        speedup = baseline_time / elapsed
        efficiency = (speedup / cores) * 100
        
        results['times'].append(elapsed)
        results['speedups'].append(speedup)
        results['efficiencies'].append(efficiency)
        
        print(f"  Time: {elapsed:.2f}s, Speedup: {speedup:.2f}x, Efficiency: {efficiency:.1f}%\n")
    
    return results, baseline_time


def plot_results(results_dict, baseline_time, output_dir='output'):
    """
    Plot scaling results.
    
    Args:
        results_dict: Dictionary of {pipeline_name: results}
        baseline_time: Sequential baseline time
        output_dir: Output directory for plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Parallel Image Processing - Core Scaling Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'd']
    
    # Plot 1: Execution Time vs Cores
    ax1 = axes[0, 0]
    for idx, (name, results) in enumerate(results_dict.items()):
        ax1.plot(results['core_counts'], results['times'], 
                marker=markers[idx % len(markers)], 
                color=colors[idx % len(colors)],
                linewidth=2, markersize=8, label=name)
    
    ax1.axhline(y=baseline_time, color='gray', linestyle='--', linewidth=1.5, label='Sequential')
    ax1.set_xlabel('Number of Cores', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Time vs Core Count', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs Cores
    ax2 = axes[0, 1]
    for idx, (name, results) in enumerate(results_dict.items()):
        ax2.plot(results['core_counts'], results['speedups'], 
                marker=markers[idx % len(markers)], 
                color=colors[idx % len(colors)],
                linewidth=2, markersize=8, label=name)
    
    # Ideal speedup (linear scaling)
    ideal_cores = results_dict[list(results_dict.keys())[0]]['core_counts']
    ax2.plot(ideal_cores, ideal_cores, 'k--', linewidth=1.5, label='Ideal (Linear)')
    
    ax2.set_xlabel('Number of Cores', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup', fontsize=11, fontweight='bold')
    ax2.set_title('Speedup vs Core Count', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency vs Cores
    ax3 = axes[1, 0]
    for idx, (name, results) in enumerate(results_dict.items()):
        ax3.plot(results['core_counts'], results['efficiencies'], 
                marker=markers[idx % len(markers)], 
                color=colors[idx % len(colors)],
                linewidth=2, markersize=8, label=name)
    
    ax3.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, label='Ideal (100%)')
    ax3.set_xlabel('Number of Cores', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Parallel Efficiency (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Parallel Efficiency vs Core Count', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bar chart comparison at max cores
    ax4 = axes[1, 1]
    names = list(results_dict.keys())
    max_core_idx = -1  # Last core count
    times_at_max = [results_dict[name]['times'][max_core_idx] for name in names]
    
    bars = ax4.bar(names, times_at_max, color=colors[:len(names)], alpha=0.7, edgecolor='black')
    ax4.axhline(y=baseline_time, color='gray', linestyle='--', linewidth=1.5, label='Sequential')
    ax4.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Performance at {results_dict[names[0]]["core_counts"][max_core_idx]} Cores', 
                 fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / 'core_scaling_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Save raw data
    data_path = output_path / 'core_scaling_data.json'
    with open(data_path, 'w') as f:
        json.dump({
            'baseline_time': baseline_time,
            'results': results_dict
        }, f, indent=2)
    print(f"Data saved to: {data_path}")
    
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Core scaling benchmark')
    parser.add_argument('--num-images', type=int, default=200, help='Number of images')
    parser.add_argument('--cores', nargs='+', type=int, default=[1, 2, 4, 8], 
                       help='Core counts to test')
    parser.add_argument('--pipelines', nargs='+', 
                       default=['processpool', 'multiprocessing_pool'],
                       choices=['processpool', 'multiprocessing_pool', 'threadpool'],
                       help='Pipelines to benchmark')
    parser.add_argument('--filter-type', default='basic', 
                       choices=['all', 'basic', 'intensive'],
                       help='Filter type')
    args = parser.parse_args()
    
    # Prepare dataset
    print("Preparing dataset...")
    image_paths = prepare_dataset(num_images=args.num_images, output_dir='./data')
    print(f"Loaded {len(image_paths)} images\n")
    
    # Filter config
    if args.filter_type == 'all':
        filters_config = {
            'grayscale': {'enabled': True},
            'gaussian_blur': {'enabled': True, 'sigma': 1.0},
            'edge_detection': {'enabled': True},
            'sharpening': {'enabled': True, 'strength': 1.5},
            'brightness': {'enabled': True, 'delta': 20},
        }
    elif args.filter_type == 'basic':
        filters_config = {
            'grayscale': {'enabled': True},
            'gaussian_blur': {'enabled': True, 'sigma': 1.0},
            'brightness': {'enabled': True, 'delta': 20},
        }
    else:  # intensive
        filters_config = {
            'edge_detection': {'enabled': True},
            'sharpening': {'enabled': True, 'strength': 2.0},
        }
    
    # Run benchmarks
    results_dict = {}
    baseline_time = None
    
    for pipeline in args.pipelines:
        results, base_time = run_benchmark(
            image_paths, 
            filters_config, 
            args.cores,
            pipeline_type=pipeline
        )
        results_dict[pipeline] = results
        if baseline_time is None:
            baseline_time = base_time
    
    # Plot results
    plot_results(results_dict, baseline_time)


if __name__ == '__main__':
    main()
