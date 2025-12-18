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
│   └── sample_images/              # Generated sample images for testing
└── output/                         # Processed images and results
    ├── processpool_samples/        # Sample outputs from ProcessPool
    └── performance_results.json    # Performance comparison results
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
- **MultiprocessingPipeline**: Direct process management with Queue
- **MultiprocessingPoolPipeline**: Uses multiprocessing.Pool for easier management
- Batch processing support
- True parallelism (bypasses Python GIL)

#### 2. concurrent.futures
- **ThreadPoolExecutor**: Suitable for I/O-bound operations
- **ProcessPoolExecutor**: True parallelism for CPU-intensive operations
- **AdaptivePipeline**: Automatically chooses executor based on workload

### Performance Features
- Automatic CPU count detection
- Progress tracking and reporting
- Performance comparison framework
- Sample image generation for testing
- Detailed performance metrics and speedup calculation

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
# Process specific number of images
python main.py --num-images 100

# Use different filter types
python main.py --filter-type intensive
# Options: 'all', 'basic', 'intensive'

# Specify output directory
python main.py --output-dir ./results

# Skip sequential baseline (faster testing)
python main.py --skip-sequential

# Test single pipeline
python main.py --test-single --pipeline processpool

# Run specific pipeline with 200 images
python main.py --test-single --pipeline multiprocessing --num-images 200
```

### Command-Line Arguments

- `--num-images NUM`: Number of images to process (default: 50)
- `--filter-type TYPE`: Type of filters ('all', 'basic', 'intensive'; default: 'basic')
- `--output-dir DIR`: Output directory (default: ./output)
- `--skip-sequential`: Skip sequential baseline comparison
- `--test-single`: Run single pipeline instead of full comparison
- `--pipeline NAME`: Pipeline to test with --test-single
  - Options: 'sequential', 'multiprocessing', 'threadpool', 'processpool'

## Implementation Details

### Filters Module (`src/filters.py`)
- **ImageFilters class**: Static methods for each filter operation
- **ImageProcessingPipeline class**: Applies sequence of filters to images

### Multiprocessing (`src/multiprocessing_pipeline.py`)
- Process-based parallelism with true parallel execution
- Queue-based result collection
- Pool-based batch processing
- Automatic worker management

### Concurrent.futures (`src/concurrent_pipeline.py`)
- High-level interface for parallel execution
- ThreadPoolExecutor for I/O-bound tasks
- ProcessPoolExecutor for CPU-bound tasks
- Adaptive strategy selection
- Future-based result handling

### Data Loader (`src/data_loader.py`)
- Generates synthetic sample images for testing
- Validates image integrity
- Provides image metadata

## Performance Metrics

The system reports:
- **Execution time** for each approach
- **Speedup** relative to sequential baseline
- **Efficiency** (speedup / number of workers × 100%)
- **JSON report** with detailed performance data

### Typical Performance Results

Results vary based on:
- Number of CPUs/cores available
- Image resolution and size
- Type of filters applied
- I/O performance
- System load

### Expected Findings

1. **Sequential**: Baseline (1x speedup)
2. **Multiprocessing**: Good speedup (2-8x on multi-core systems)
3. **ThreadPool**: Lower speedup (affected by Python GIL)
4. **ProcessPool**: Best speedup for CPU-intensive operations (2-8x)

## Python GIL Implications

- **ThreadPoolExecutor**: Limited by Global Interpreter Lock (GIL) - good for I/O
- **Multiprocessing & ProcessPoolExecutor**: Bypass GIL with separate processes - best for CPU-intensive image processing

## Output

### Generated Files

1. **performance_results.json**: Detailed performance metrics
2. **processpool_samples/**: Sample processed images
3. Console output with progress and timing information

## Sample Output

```
============================================================
PERFORMANCE COMPARISON SUMMARY
============================================================

Method                         Time (s)     Speedup    Efficiency  
------------------------------------------------------------
sequential                     45.23        1.00x      100.0%
multiprocessing_pool           8.94         5.06x      63.2%
processpool                     7.23         6.26x      78.3%
threadpool                      38.45        1.18x      14.8%
============================================================
```

## Troubleshooting

### Issue: "No module named 'cv2'"
**Solution**: Install opencv-python:
```bash
pip install opencv-python
```

### Issue: Memory issues with large datasets
**Solution**: Reduce `--num-images` or use smaller `--filter-type`:
```bash
python main.py --num-images 20 --filter-type basic
```

### Issue: Slow performance on single-core systems
**Solution**: The system is optimized for multi-core systems. Results on single-core will be slower.

## Educational Objectives

This assignment demonstrates:
1. ✓ Different parallel programming paradigms in Python
2. ✓ Trade-offs between multiprocessing and threading
3. ✓ Performance comparison and benchmarking
4. ✓ Image processing fundamentals
5. ✓ Scalability considerations
6. ✓ Resource management in parallel systems

## Notes

- Sample images are auto-generated if not present
- To use real Food-101 dataset images, download from Kaggle and place in `data/food-101/`
- All image operations maintain 8-bit color depth (0-255)
- Filters are applied sequentially in the order: grayscale → blur → edge → sharpen → brightness

## Future Enhancements

Possible additions for the assignment:
1. GPU acceleration using CUDA/CuPy
2. Distributed processing with Dask
3. Cloud deployment on GCP
4. Advanced profiling and optimization
5. Real-time streaming pipeline
6. Web API for image processing service

## References

- Python multiprocessing: https://docs.python.org/3/library/multiprocessing.html
- concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html
- OpenCV documentation: https://docs.opencv.org/
- Food-101 Dataset: https://www.kaggle.com/datasets/dansbecker/food-101

## License

Academic use only - CST435 Assignment

## Author

CST435 Student
