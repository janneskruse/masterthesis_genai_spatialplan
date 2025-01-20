## Repository accompanying master thesis on generative AI in spatial planning


### connect to jupyter hub server on HPC
- selet exisitng juypter hub server in vs code on kernel selection
- create a new session in jupyter hub
- generate a token in the jupyter hub session
- set url to `https://lab.sc.uni-leipzig.de/jupyter/`
- set username to hpc username
- set password/token to generated token


### Installation
- install from conda environment.yml: `conda env create -f environment.yml`
- activate conda environment: `conda activate genaiSpatialplan`
- install jupyter kernel using: `python -m ipykernel install --user --name genaiSpatialplan`
- update environment.yml: `conda env export --name genaiSpatialplan --file environment.yml`