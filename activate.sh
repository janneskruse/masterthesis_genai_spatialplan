#!/bin/bash
# Quick activation script - source this file
# Usage: source activate.sh (or: . activate.sh)

conda env update -f environment.yml --prune
conda run -n genaiSpatialplan pip install -e . --no-deps
conda activate genaiSpatialplan

echo "✓ Environment activated!"

conda info --envs | grep "\*"
