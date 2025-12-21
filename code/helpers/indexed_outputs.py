"""Helper functions for managing indexed output files."""

import os
import re
from pathlib import Path
from typing import Optional


def get_next_run_idx(output_dir: str, base_name: str) -> int:
    """
    Find the next available run index for output files.
    
    Scans the output directory for files matching the pattern:
    {base_name}_*_idx{number}.{ext} or {base_name}_idx{number}.{ext}
    
    Args:
        output_dir: Directory to search for existing files
        base_name: Base name of the output files (e.g., 'samples_guidance7.5')
        
    Returns:
        Next available index (max existing index + 1, or 0 if none exist)
        
    Examples:
        >>> get_next_run_idx('/path/to/outputs', 'samples_guidance7.5')
        # If files exist: samples_guidance7.5_idx0.png, samples_guidance7.5_idx1.png
        # Returns: 2
        
        >>> get_next_run_idx('/path/to/outputs', 'sample')
        # If files exist: sample_0_idx3.png, sample_1_idx3.png
        # Returns: 4
    """
    if not os.path.exists(output_dir):
        return 0
    
    # Pattern to match files with idx suffix before extension
    # Matches: {base_name}*_idx{number}.{ext}
    pattern = re.compile(rf'{re.escape(base_name)}.*_idx(\d+)\.')
    
    max_idx = -1
    
    # Scan all files in directory
    for filename in os.listdir(output_dir):
        match = pattern.search(filename)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)
    
    # Return next index (0 if no files found)
    return max_idx + 1
