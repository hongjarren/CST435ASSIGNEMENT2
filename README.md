# Parallel Image Processing Assignment - CST435

A complete implementation of a parallel image processing system using different parallelization paradigms in Python.

## Project Overview

This project implements a Food-101 dataset image processor that applies various filters using:
- **multiprocessing** module
- **concurrent.futures** (ThreadPoolExecutor and ProcessPoolExecutor)

## Project Structure

```
CST435 Assignment 2/
├── main.py                          # Main entry point and performance comparison
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── src/
│   ├── filters.py                  # Image filter implementations
│   ├── data_loader.py              # Food-101 dataset loader
│   ├── multiprocessing_pipeline.py # Multiprocessing implementations
│   └── concurrent_pipeline.py      # concurrent.futures implementations
├── data/                           # Dataset directory
│   ├── food-101/                   # Full Food-101 dataset (if downloaded)
│   ├── my_subset/                  # Generated sample subset for testing
│   └── subset_2000/                # Larger sample subset
└── output/                         # Processed images and results
    ├── benchmark/                  # Benchmark-specific outputs
    ├── processpool_samples/        # Sample outputs from ProcessPool
    ├── performance_results.json    # Performance comparison results
    ├── core_scaling_results.json   # Core scaling benchmark data
    ├── core_scaling_data.json      # Raw core scaling data
    └── core_scaling_plots.png      # Visualization of scaling performance
```

## Features

### Image Filters Implemented

1. **Grayscale Conversion** - Converts RGB to grayscale using luminance formula
   - Formula: Gray = 0.299*R + 0.587*G + 0.114*B

2. **Gaussian Blur** - Applies 3×3 Gaussian kernel for smoothing
   - Configurable sigma parameter for blur intensity

3. **Edge Detection** - Sobel filter for edge detection
   - Computes X and Y gradients
   - Returns edge magnitude

4. **Image Sharpening** - Enhances edges and details
   - Uses unsharp masking technique
   - Configurable strength parameter

5. **Brightness Adjustment** - Increases or decreases image brightness
   - Configurable delta parameter (-127 to 127)

### Parallel Implementations

#### 1. Multiprocessing Module
- **MultiprocessingPoolPipeline**: Uses multiprocessing.Pool for process management
- Batch processing support
- True parallelism (bypasses Python GIL)
- Optimal for CPU-intensive image processing tasks

#### 2. concurrent.futures
- **ThreadPoolPipeline**: Thread-based parallelism using ThreadPoolExecutor
- **ProcessPoolPipeline**: Process-based parallelism using ProcessPoolExecutor
- High-level interface with future-based result handling
- Best performance with ProcessPoolExecutor for CPU-bound operations

### Performance Features
- Automatic CPU count detection
- Progress tracking and reporting
- Performance comparison framework
- **Core scaling benchmark** - Test performance across different CPU core counts
- Sample image generation for testing
- Detailed performance metrics and speedup calculation
- Visualization plots for core scaling analysis
- JSON export for all benchmark results

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run full performance comparison with default settings:
```bash
python main.py
```

### Advanced Options

```bash
# Run core scaling benchmark
python main.py --benchmark-cores --num-images 100

# Custom core counts for benchmark
python main.py --benchmark-cores --core-counts 1 2 4 6 8 --num-images 50

# Specify number of workers
python main.py --num-workers 4 --num-images 100
```

## Implementation Details

### Filters Module (`src/filters.py`)
- **ImageFilters class**: Static methods for each filter operation
- **ImageProcessingPipeline class**: Applies sequence of filters to images
- Supports configurable filter parameters

### Multiprocessing (`src/multiprocessing_pipeline.py`)
- **MultiprocessingPoolPipeline**: Process-based parallelism using multiprocessing.Pool
- True parallel execution (bypasses GIL)
- Automatic worker management based on CPU count
- Batch processing with progress tracking

### Concurrent.futures (`src/concurrent_pipeline.py`)
- **ConcurrentPipeline**: Base class for concurrent execution
- **ThreadPoolPipeline**: ThreadPoolExecutor for thread-based parallelism
- **ProcessPoolPipeline**: ProcessPoolExecutor for process-based parallelism
- Future-based result handling with error management
- Progress tracking and verbose output options

### Data Loader (`src/data_loader.py`)
- Generates synthetic sample images for testing
- Validates image integrity
- Provides image metadata

## Output

### Generated Files

1. **performance_results.json**: Detailed performance comparison metrics
2. **core_scaling_results.json**: Core scaling benchmark results
3. **core_scaling_data.json**: Raw core scaling benchmark data
4. **core_scaling_plots.png**: Visualization of scaling performance
5. **core_scaling_analysis.png**: Additional scaling analysis plots
6. **processpool_samples/**: Sample processed images from ProcessPool
7. **benchmark/**: Additional benchmark-specific outputs
8. Console output with progress and timing information

## Sample Output



```
============================================================
CORE SCALING BENCHMARK
============================================================
Filter type: basic
Images: 100

Cores    MultiPool(s)  ThreadPool(s)  ProcessPool(s)
------------------------------------------------------------
1        89.45         95.12          88.73
2        47.23         78.45          46.89
4        25.67         65.23          24.12
8        15.89         58.91          14.45
============================================================

Results saved to: output/core_scaling_results.json
Plots saved to: output/core_scaling_plots.png
```