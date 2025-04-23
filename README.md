## Repository accompanying master thesis on generative AI in spatial planning


### connect to hpc shell + useful commands
- ssh login using: `ssh <username>@login01.sc.uni-leipzig.de`
- check available modules: `module avail`
- load required modules: `module load <module_name>`

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
- update environment: `conda env update --name genaiSpatialplan --file environment.yml`

Deprecated:
- if you can not install python >=3.10 on your system: install from conda environment.yml: `conda env create -f environment.yml`
    - then activate conda environment: `conda activate genaiSpatialplan`
- else `conda deactivate` if you have conda installed
- install poetry: `poetry install` to install packages from pyproject.toml
- create jupyter kernel using: `poetry run python -m ipykernel install --user --name tf-genaiSpatialplan`