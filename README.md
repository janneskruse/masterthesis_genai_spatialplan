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

### Copy from and to HPC
- copy files from local to HPC: `scp <local_file_path> <username>@login01.sc.uni-leipzig.de:<remote_file_path>`
- copy files from HPC to local: `scp <username>@login01.sc.uni-leipzig.de:<remote_file_path> <local_file_path>`

Steps:
1. create a new workspace on the HPC cluster: `ws_allocate <name> <duration>`, e.g. `ws_allocate genai_spatial 30`
2. Set a reminder email before a workspace expires with `ws_send_ical <workspace~name> "<your~email>", e.g. `ws_send_ical <username>-genai_spatial "<your_email>@example.com"`
3. Convert dataset to zip
4. Copy the local dataset to the HPC cluster using `scp` command. e.g for model_input_dataset.zarr `scp model_input_dataset.zarr.zip <username>@login01.sc.uni-leipzig.de:/work/<username>-genai_spatial`
5. Unzip the dataset on the HPC cluster using `unzip model_input_dataset.zarr.zip`