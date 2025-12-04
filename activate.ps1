# Quick activation for Windows
# Run this then manually activate, or add as function to $PROFILE

# conda env update -f environment.yml --prune
conda run -n genaiSpatialplan pip install -e . --no-deps
conda activate genaiSpatialplan

Write-Host ""
Write-Host "✓ Environment ready! Check if activated:" -ForegroundColor Green

conda info --envs | Select-String "\*"
