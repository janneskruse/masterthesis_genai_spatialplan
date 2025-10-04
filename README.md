## Repository accompanying master thesis on generative AI in spatial planning


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

### Running the pipeline
There is a pipeline for the data acquisition and processing to acquire the model input dataset. The pipeline can be configured to choose different regions and different temperature settings using the [config.yml](./config.yml) file. 

To run the pipeline:

1. Please ensure all parameters are set before running the pipeline. To check which regions are available, open [data/ghsl/ghsl_data.parquet](./data/ghsl/ghsl_data.parquet) - e.g. using pandas or geopandas.
2. Create a new workspace under your username on the HPC cluster. You can do this by running `ws_allocate <name> <duration>` in the HPC shell, e.g. `ws_allocate master 30` to create a workspace named `<username>-master` for 30 days.
3. Make sure to download and convert the respective building height dataset like done and explained in the notebook [osm_to_xarray.ipynb](./code/data_acquisition/osm_to_xarray.ipynb). For Germany there already is a parquet file containing the building height data [here](https://www.dropbox.com/scl/fi/g1krcq2zj5wb6letsf65m/building_heights_germany.parquet?rlkey=a8pmpqtlu9wowttvfxgcb5rjp&st=twctw6j3&dl=0) that you can download and save to [data/che_etal/Germany_Hungary_Iceland](./data/che_etal/Germany_Hungary_Iceland) for the pipeline to work on all German regions.
4. Create the conda environment like indicated above and activate it.
5. Submit the pipeline to the HPC cluster using the [`submit_pipeline.py`](./code/data_acquisition/slurm/submit_pipeline.py) script: `python submit_pipeline.py`. This script will automatically create jobs for all pipeline steps. To check the status, run `squeue -u <username>` on the HPC cluster.

### Running the scripts standalone
1. Run the osm_to_xarray script with the region as environment variable using: 
```powershell
$env:REGION = "your_region" 
python ./code/data_acquisition/slurm/osm_to_xarray.py
```
2. Run the landsat_to_xarray script with the region as environment variable using: 
```powershell
$env:REGION = "your_region" 
python ./code/data_acquisition/slurm/landsat_to_xarray.py
```
3. Run the planetscope_to_xarray notebook
4. Lastly, use the combine_datasets notebook to combine all datasets to the final model input dataset.

### Copy from and to HPC
- copy files from local to HPC: `scp <local_file_path> <username>@login01.sc.uni-leipzig.de:<remote_file_path>`
- copy files from HPC to local: `scp <username>@login01.sc.uni-leipzig.de:<remote_file_path> <local_file_path>`

Steps:
1. create a new workspace on the HPC cluster: `ws_allocate <name> <duration>`, e.g. `ws_allocate genai_spatial 30`
2. Set a reminder email before a workspace expires with `ws_send_ical <workspace~name> "<your~email>", e.g. `ws_send_ical <username>-genai_spatial "<your_email>@example.com"`
3. Convert dataset to zip
4. Copy the local dataset to the HPC cluster using `scp` command. e.g for model_input_dataset.zarr `scp model_input_dataset.zarr.zip <username>@login01.sc.uni-leipzig.de:/work/<username>-genai_spatial`
5. Unzip the dataset on the HPC cluster using `unzip model_input_dataset.zarr.zip`

### connect to hpc shell + useful commands
- ssh login using: `ssh <username>@login01.sc.uni-leipzig.de`
- check available modules: `module avail`
- load required modules: `module load <module_name>`

### connect to jupyter hub server on HPC
- select existing jupyter hub server in vs code on kernel selection
- create a new session in jupyter hub
- generate a token in the jupyter hub session
- set url to `https://lab.sc.uni-leipzig.de/jupyter/`
- set username to hpc username
- set password/token to generated token