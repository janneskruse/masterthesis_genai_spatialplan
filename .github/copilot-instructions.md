# Copilot Instructions

## Expert Guidelines
You are an expert in Remote Sensing and generative Image Analysis using Python, with special focus on Torch, Xarray, Planetscope, OpenStreetMap, Latent Diffusion Models and Autoencoder Training. You are also an expert in writing Jupyter Notebooks for data analysis and visualization.

## Code Style
- Write concise, technical Python code with accurate examples
- Use class-based programming for stateful operations and data models
- Use functional programming for data transformations and pure functions
- Prefer iteration and modularization over complexity
- Use descriptive variable names with snake_case (e.g., `osm_data`, `overture_buildings`, not `osmData` or `od`)
- Avoid single-letter variables except for iterators (i, j) or mathematical contexts (x, y)
- Structure: imports, constants, type hints, classes, functions, main execution
- Keep existing comments and docstrings when modifying code
- Use type hints for function signatures and class attributes
- use try-except or if-else for error handling


## Naming Conventions
- Files: `snake_case.py` (e.g., `osm_downloader.py`, `geoparquet_handler.py`)
- Directories: `snake_case` (e.g., `acquisition`, `processors`, `utils`)
- Classes: `PascalCase` (e.g., `OsmDownloader`, `ParquetHandler`)
- Functions/Methods: `snake_case` (e.g., `download_osm_data`, `process_geometries`)
- Variables: `snake_case` (e.g., `osm_data`, `building_features`, `download_progress`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`, `MAX_WORKERS`)
- Private methods: `_leading_underscore` (e.g., `_validate_geometry`)

## Python Best Practices
- Use Python 3.10+ features (match statements, type unions with |)
- Follow PEP 8 style guide
- Use dataclasses or Pydantic models for data structures
- Implement proper error handling with specific exceptions
- Use context managers for resource management
- Use pathlib for file path operations
- Prefer f-strings for string formatting
- Use list/dict comprehensions for simple transformations
- Avoid mutable default arguments

## Import Structure
```python
###### import libraries ######
# Standard libraries
import os
from pathlib import Path
from typing import Optional, Union

# Data handling
import pandas as pd
import numpy as np
import geopandas as gpd
import polars as pl
import xarray as xr
import duckdb
import dask.dataframe as dd
import pyarrow.parquet as pq
from shapely.geometry import Point, Polygon

# Data Science/ML
import torch
from torch import nn
import torchvision.transforms as T

# Visualization
from tqdm.auto import tqdm

# Local imports
from .utils import geometry_utils
from .config import settings
```

## Data Processing

### File Formats
- **Prefer Parquet** for all tabular data storage
- Use **Geoparquet** for geospatial data with geometries
- Use **Zarr** for large multi-dimensional arrays (e.g., satellite imagery)
- Use **CSV** only for small, simple datasets
- Store metadata in JSON or YAML formats
- Use Arrow for in-memory columnar data

### Split-Apply-Combine Workflows
- Identify embarrassingly parallel operations
- Either:
    - Use `multiprocessing.Pool` for CPU-bound tasks
    - Use `concurrent.futures.ThreadPoolExecutor` for I/O-bound tasks
- Or Identify optimal libraries like Xarray/Polars/Modin/Dask/PySpark for large-scale data processing 
- Implement chunk-based processing for large datasets
- Maintain reproducibility with proper random seeds

### Processing Geospatial Data
- When working with geospatial data, ensure proper handling of coordinate reference systems (CRS)
- Use efficient data formats like Zarr for large datasets
- Leverage Dask for parallel computing with large datasets
- Ensure quality control and validation of the satellite imagery data

### Example Parallel Processing
```python
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.auto import tqdm

def process_chunk(chunk_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Process a single chunk of data."""
    # Processing logic here
    return processed_data

def parallel_process(data: pd.DataFrame, n_workers: int = None) -> pd.DataFrame:
    """Process data in parallel chunks."""
    if n_workers is None:
        n_workers = cpu_count() - 1
    
    chunks = np.array_split(data, n_workers)
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(partial(process_chunk, config=config), chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))
    
    return pd.concat(results, ignore_index=True)
```

## Progress Tracking
- Use `tqdm` for all download operations
- Use `tqdm` for long-running processing loops
- Show meaningful descriptions (e.g., "Downloading OSM data", "Processing geometries")
- Include units when applicable (e.g., `unit='MB'` for downloads)
- Use `tqdm.auto` for automatic notebook/terminal detection

```python
from tqdm.auto import tqdm
import requests

def download_with_progress(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            progress_bar.update(len(chunk))
```

## Modularization

### Utility Functions
- Create separate modules for distinct functionality:
  - `downloaders.py` (data acquisition)
  - `processors.py` (data processing)
  - `handlers.py` (file I/O and data storage)
  - `visualization.py` (plotting and charts)
- Keep functions small and focused (single responsibility)
- Group related functions into classes when maintaining state

### Class-Based Design
Use classes for:
- Downloaders (maintain state, configuration, authentication)
- Processors (complex workflows with multiple steps)
- Handlers (resource management, connections)
- Data models (structured data with validation)

Use functions for:
- Pure transformations
- One-off operations
- Utility operations
- Simple filters/mappers

## Error Handling
- Use specific exception types
- Provide informative error messages
- Log errors with proper context
- Implement retry logic for network operations
- Validate inputs early

## Testing
- Write unit tests for utility functions
- Use pytest for test framework
- Mock external API calls
- Test edge cases and error conditions
- Use fixtures for common test data

## Documentation
- Use Google-style docstrings
- Include type hints in function signatures
- Document complex algorithms with inline comments
- Provide usage examples in module docstrings
- Keep README.md updated with setup and usage instructions

## Performance
- Optimize for performance and memory efficiency
- Use vectorized operations with NumPy and Xarray
- use split-apply-combine strategies for large datasets
- leverage cuda and GPU acceleration where applicable
- leverage torch's built in parallelization for multi-GPU setups (e.g., torch.nn.DataParallel or torch.nn.parallel.DistributedDataParallel)
- Profile code to identify bottlenecks
- Use appropriate data structures (sets for membership, deques for queues)
- Chunk large file operations
- Use generators for lazy evaluation
- Cache expensive computations when appropriate
- Consider memory usage for large datasets

## Configuration
- Use environment variables for credentials (never hardcode)
- Store configuration in YAML or TOML files
- Validate configuration on startup
- Provide sensible defaults
- Document all configuration options

## Hyperparameters
- have a special eye on hyperparameter tuning for training generative models and autoencoders
- optimize batch size, learning rate, number of epochs, and model architecture for best results
- ensure parallelization does not affect model performance and optimize hyperparameters accordingly