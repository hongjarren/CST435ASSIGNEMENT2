"""
Main Script - Parallel Image Processing Assignment
Demonstrates and compares different parallelization approaches
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import prepare_dataset, Food101DataLoader
from filters import ImageProcessingPipeline
from multiprocessing_pipeline import MultiprocessingPoolPipeline
from concurrent_pipeline import ThreadPoolPipeline, ProcessPoolPipeline


class PerformanceComparison:
    """Compare performance of different parallelization approaches"""
    
    def __init__(self, output_dir: str = "./output", num_workers: int = None):
        """
        Initialize performance comparison.
        
        Args:
            output_dir: Directory to save results and processed images
            num_workers: Number of workers for parallel execution (None = optimal/CPU count)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.results = {}
    
    def get_filter_config(self, filter_type: str = "all") -> Dict:
        """
        Get filter configuration for testing.
        
        Args:
            filter_type: Type of filters to apply ('all', 'basic', 'intensive')
            
        Returns:
            Filter configuration dictionary
        """
        if filter_type == "all":
            return {
                'grayscale': {'enabled': True},
                'gaussian_blur': {'enabled': True, 'sigma': 1.0},
                'edge_detection': {'enabled': True},
                'sharpening': {'enabled': True, 'strength': 1.5},
                'brightness': {'enabled': True, 'delta': 20},
            }
        elif filter_type == "basic":
            return {
                'grayscale': {'enabled': True},
                'gaussian_blur': {'enabled': True, 'sigma': 1.0},
                'brightness': {'enabled': True, 'delta': 20},
            }
        else:  # intensive
            return {
                'edge_detection': {'enabled': True},
                'sharpening': {'enabled': True, 'strength': 2.0},
            }
    
    def run_sequential(self, image_paths: List[str], filters_config: Dict) -> Tuple[float, List]:
        """
        Process images sequentially (baseline).
        
        Args:
            image_paths: List of image paths
            filters_config: Filter configuration
            
        Returns:
            Tuple of (execution_time, results)
        """
        print("\n" + "="*60)
        print("SEQUENTIAL PROCESSING (Baseline)")
        print("="*60)
        
        pipeline = ImageProcessingPipeline()
        results = []
        
        start_time = time.time()
        
        for idx, image_path in enumerate(image_paths):
            try:
                _, processed_image = pipeline.process_image(image_path, filters_config)
                results.append((image_path, processed_image))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(image_paths)} images")
        
        elapsed_time = time.time() - start_time
        
        print(f"Sequential processing completed in {elapsed_time:.2f} seconds")
        print(f"Successfully processed {len(results)} images")
        
        return elapsed_time, results
    
    def run_multiprocessing_pool(self, image_paths: List[str], filters_config: Dict) -> Tuple[float, List]:
        """
        Process images using multiprocessing.Pool.
        
        Args:
            image_paths: List of image paths
            filters_config: Filter configuration
            
        Returns:
            Tuple of (execution_time, results)
        """
        print("\n" + "="*60)
        print(f"MULTIPROCESSING POOL PIPELINE (Workers: {self.num_workers or 'CPU count'})")
        print("="*60)
        
        pipeline = MultiprocessingPoolPipeline(num_processes=self.num_workers)
        
        start_time = time.time()
        results = pipeline.process_images(image_paths, filters_config, verbose=True)
        elapsed_time = time.time() - start_time
        
        print(f"Multiprocessing Pool completed in {elapsed_time:.2f} seconds")
        
        return elapsed_time, results
    
    def run_threadpool(self, image_paths: List[str], filters_config: Dict) -> Tuple[float, List]:
        """
        Process images using ThreadPoolExecutor.
        
        Args:
            image_paths: List of image paths
            filters_config: Filter configuration
            
        Returns:
            Tuple of (execution_time, results)
        """
        print("\n" + "="*60)
        print(f"THREADPOOL EXECUTOR PIPELINE (Workers: {self.num_workers or 'optimal'})")
        print("="*60)
        
        pipeline = ThreadPoolPipeline(num_workers=self.num_workers)
        
        start_time = time.time()
        results = pipeline.process_images(image_paths, filters_config, verbose=True)
        elapsed_time = time.time() - start_time
        
        print(f"ThreadPoolExecutor completed in {elapsed_time:.2f} seconds")
        
        return elapsed_time, results
    
    def run_processpool(self, image_paths: List[str], filters_config: Dict) -> Tuple[float, List]:
        """
        Process images using ProcessPoolExecutor.
        
        Args:
            image_paths: List of image paths
            filters_config: Filter configuration
            
        Returns:
            Tuple of (execution_time, results)
        """
        print("\n" + "="*60)
        print(f"PROCESS POOL EXECUTOR PIPELINE (Workers: {self.num_workers or 'CPU count'})")
        print("="*60)
        
        pipeline = ProcessPoolPipeline(num_workers=self.num_workers)
        
        start_time = time.time()
        results = pipeline.process_images(image_paths, filters_config, verbose=True)
        elapsed_time = time.time() - start_time
        
        print(f"ProcessPoolExecutor completed in {elapsed_time:.2f} seconds")
        
        return elapsed_time, results
    
   
    
    def save_sample_results(self, results: List[Tuple[str, np.ndarray]], 
                           pipeline_name: str, max_samples: int = 5):
        """
        Save sample processed images.
        
        Args:
            results: List of (image_path, processed_image) tuples
            pipeline_name: Name of the pipeline for directory naming
            max_samples: Maximum number of samples to save
        """
        output_subdir = self.output_dir / pipeline_name
        output_subdir.mkdir(exist_ok=True)
        
        for idx, (original_path, processed_image) in enumerate(results[:max_samples]):
            # Save processed image
            filename = Path(original_path).stem + "_processed.jpg"
            output_path = output_subdir / filename
            
            # Convert RGB to BGR for cv2
            if len(processed_image.shape) == 3:
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            else:
                processed_image_bgr = processed_image
            
            cv2.imwrite(str(output_path), processed_image_bgr)
    
    def print_comparison_summary(self):
        """Print performance comparison summary"""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*60)
        
        if not self.results:
            print("No results to compare")
            return
        
        # Sort by execution time
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['time'])
        
        # Find baseline (sequential)
        baseline_time = self.results.get('sequential', {}).get('time', 1.0)
        
        print(f"\n{'Method':<30} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<10}")
        print("-" * 62)
        
        for method, data in sorted_results:
            exec_time = data['time']
            speedup = baseline_time / exec_time
            num_processes = data.get('num_workers', 1)
            
            # Handle 'optimal' num_processes
            if self.num_workers is not None:
                num_processes = self.num_workers
            elif isinstance(num_processes, str):
                num_processes = 1
            
            efficiency = (speedup / num_processes * 100) if num_processes > 0 else 0
            
            print(f"{method:<30} {exec_time:<12.2f} {speedup:<10.2f}x {efficiency:<10.1f}%")
        
        print("-" * 62)
        
        # Save results to JSON
        results_file = self.output_dir / "performance_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to {results_file}")
    
    def run_all_comparisons(self, image_paths: List[str], 
                           filter_type: str = "all",
                           skip_sequential: bool = False):
        """
        Run all pipeline comparisons.
        
        Args:
            image_paths: List of image paths
            filter_type: Type of filters to apply
            skip_sequential: Skip sequential baseline
        """
        filters_config = self.get_filter_config(filter_type)
        
        print(f"\n{'*'*60}")
        print(f"STARTING PERFORMANCE COMPARISON")
        print(f"Images: {len(image_paths)}")
        print(f"Filter Type: {filter_type}")
        print(f"{'*'*60}")
        
        # Sequential baseline
        if not skip_sequential:
            exec_time, _ = self.run_sequential(image_paths, filters_config)
            self.results['sequential'] = {
                'time': exec_time,
                'num_workers': 1,
                'method': 'Sequential Processing'
            }
        
        # Multiprocessing Pool
        try:
            exec_time, _ = self.run_multiprocessing_pool(image_paths, filters_config)
            import os
            self.results['multiprocessing_pool'] = {
                'time': exec_time,
                'num_workers': self.num_workers or os.cpu_count(),
                'method': 'Multiprocessing Pool'
            }
        except Exception as e:
            print(f"Multiprocessing Pool failed: {e}")
        
        # ThreadPool
        try:
            exec_time, _ = self.run_threadpool(image_paths, filters_config)
            self.results['threadpool'] = {
                'time': exec_time,
                'num_workers': self.num_workers or 'optimal',
                'method': 'ThreadPoolExecutor'
            }
        except Exception as e:
            print(f"ThreadPool failed: {e}")
        
        # ProcessPool
        try:
            exec_time, results = self.run_processpool(image_paths, filters_config)
            import os
            self.results['processpool'] = {
                'time': exec_time,
                'num_workers': self.num_workers or os.cpu_count(),
                'method': 'ProcessPoolExecutor'
            }
            # Save sample results
            self.save_sample_results(results, 'processpool_samples')
        except Exception as e:
            print(f"ProcessPool failed: {e}")
        
        # Print summary
        self.print_comparison_summary()
    
    def run_core_scaling_benchmark(self, image_paths: List[str], filter_type: str, core_counts: List[int]):
        """
        Run benchmark across multiple core counts for all parallel pipelines.
        
        Args:
            image_paths: List of image file paths
            filter_type: Type of filters to apply
            core_counts: List of core counts to test
        """
        import os
        filters_config = self.get_filter_config(filter_type)
        
        # Run sequential baseline once
        print("\n" + "="*70)
        print("CORE SCALING BENCHMARK")
        print("="*70)
        print(f"\nImages: {len(image_paths)}")
        print(f"Filters: {filter_type}")
        print(f"Core counts: {core_counts}")
        print("\nRunning sequential baseline...")
        baseline_time, _ = self.run_sequential(image_paths, filters_config)
        print(f"Sequential baseline: {baseline_time:.2f}s\n")
        
        # Results storage
        results = {
            'baseline': baseline_time,
            'pipelines': {}
        }
        
        pipelines = [
            ('multiprocessing_pool', 'Multiprocessing Pool', self.run_multiprocessing_pool),
            ('processpool', 'ProcessPoolExecutor', self.run_processpool),
            ('threadpool', 'ThreadPoolExecutor', self.run_threadpool)
        ]
        
        for pipeline_key, pipeline_name, pipeline_func in pipelines:
            print(f"\n{'='*70}")
            print(f"Testing: {pipeline_name}")
            print(f"{'='*70}")
            
            pipeline_results = []
            for cores in core_counts:
                print(f"\nTesting with {cores} cores...")
                self.num_workers = cores
                try:
                    exec_time, _ = pipeline_func(image_paths, filters_config)
                    speedup = baseline_time / exec_time
                    efficiency = (speedup / cores) * 100
                    
                    pipeline_results.append({
                        'cores': cores,
                        'time': exec_time,
                        'speedup': speedup,
                        'efficiency': efficiency
                    })
                    
                    print(f"  Time: {exec_time:.2f}s")
                    print(f"  Speedup: {speedup:.2f}x")
                    print(f"  Efficiency: {efficiency:.1f}%")
                except Exception as e:
                    print(f"  Failed: {e}")
            
            results['pipelines'][pipeline_key] = {
                'name': pipeline_name,
                'results': pipeline_results
            }
        
        # Print summary table
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\nSequential Baseline: {baseline_time:.2f}s\n")
        
        for pipeline_key, pipeline_data in results['pipelines'].items():
            print(f"\n{pipeline_data['name']}:")
            print(f"  {'Cores':<8} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12}")
            print(f"  {'-'*44}")
            for r in pipeline_data['results']:
                print(f"  {r['cores']:<8} {r['time']:<12.2f} {r['speedup']:<12.2f}x {r['efficiency']:<12.1f}%")
        
        # Save results to JSON
        results_file = self.output_dir / 'core_scaling_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Generate plots
        self._plot_core_scaling_results(results)
        print(f"\nPlots saved to: {self.output_dir / 'core_scaling_plots.png'}")
    
    def _plot_core_scaling_results(self, results: Dict):
        """
        Generate visualization plots for core scaling benchmark results.
        
        Args:
            results: Dictionary containing benchmark results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Core Scaling Performance Analysis - Execution Time vs Cores', fontsize=16, fontweight='bold')
        
        colors = {
            'multiprocessing': '#2E86AB',
            'multiprocessing_pool': '#A23B72',
            'processpool': '#F18F01',
            'threadpool': '#C73E1D'
        }
        
        axes = [ax1, ax2, ax3, ax4]
        pipeline_items = list(results['pipelines'].items())
        
        # Create separate plot for each pipeline
        for idx, (pipeline_key, pipeline_data) in enumerate(pipeline_items):
            ax = axes[idx]
            cores = [r['cores'] for r in pipeline_data['results']]
            times = [r['time'] for r in pipeline_data['results']]
            
            ax.plot(cores, times, marker='o', linewidth=2.5, markersize=10,
                   color=colors.get(pipeline_key, 'gray'))
            
            # Add value labels on points
            for core, time in zip(cores, times):
                ax.text(core, time, f'{time:.2f}s', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Number of Cores', fontsize=11, fontweight='bold')
            ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
            ax.set_title(f'{pipeline_data["name"]}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(cores)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'core_scaling_plots.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Parallel Image Processing Assignment')
    parser.add_argument('--num-images', type=int, default=50,
                       help='Number of images to process (default: 50)')
    parser.add_argument('--filter-type', choices=['all', 'basic', 'intensive'], 
                       default='basic',
                       help='Type of filters to apply (default: basic)')
    parser.add_argument('--output-dir', default='./output',
                       help='Output directory for results (default: ./output)')
    parser.add_argument('--skip-sequential', action='store_true',
                       help='Skip sequential baseline (for faster testing)')
    parser.add_argument('--test-single', action='store_true',
                       help='Run single-pipeline test instead of full comparison')
    parser.add_argument('--pipeline', choices=['sequential', 'multiprocessing_pool', 'threadpool', 'processpool'],
                       default='processpool',
                       help='Specific pipeline to test (with --test-single)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of workers (threads/processes) to use (default: optimal/CPU count)')
    parser.add_argument('--benchmark-cores', action='store_true',
                       help='Run core scaling benchmark across multiple core counts')
    parser.add_argument('--core-counts', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Core counts to test in benchmark mode (default: 1 2 4 8)')
    
    args = parser.parse_args()
    
    # Prepare dataset
    print("Preparing dataset...")
    image_paths = prepare_dataset(num_images=args.num_images, output_dir='./data')
    print(f"Loaded {len(image_paths)} images")
    
    if args.benchmark_cores:
        # Run core scaling benchmark
        comparison = PerformanceComparison(output_dir=args.output_dir)
        comparison.run_core_scaling_benchmark(
            image_paths,
            filter_type=args.filter_type,
            core_counts=args.core_counts
        )
    elif args.test_single:
        # Run single pipeline test
        print(f"\nRunning single pipeline test: {args.pipeline}")
        comparison = PerformanceComparison(output_dir=args.output_dir, num_workers=args.num_workers)
        filters_config = comparison.get_filter_config(args.filter_type)
        
        if args.pipeline == 'sequential':
            exec_time, results = comparison.run_sequential(image_paths, filters_config)
        elif args.pipeline == 'multiprocessing_pool':
            exec_time, results = comparison.run_multiprocessing_pool(image_paths, filters_config)
        elif args.pipeline == 'threadpool':
            exec_time, results = comparison.run_threadpool(image_paths, filters_config)
        else:  # processpool
            exec_time, results = comparison.run_processpool(image_paths, filters_config)
        
        print(f"\nExecution time: {exec_time:.2f} seconds")
    else:
        # Run full comparison
        comparison = PerformanceComparison(output_dir=args.output_dir, num_workers=args.num_workers)
        comparison.run_all_comparisons(
            image_paths,
            filter_type=args.filter_type,
            skip_sequential=args.skip_sequential
        )


if __name__ == '__main__':
    main()
