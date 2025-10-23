# Repository accompanying the master thesis on generative AI in spatial planning

| title: A heat island aware generative urban layout model. Can artificial intelligence generate new scenarios of city areas optimized for future climates, for more informed decision making?

| author: Jannes Kruse


## Installation
- install from conda environment.yml: `conda env create -f environment.yml`
- activate conda environment: `conda activate genaiSpatialplan`
- install jupyter kernel using: `python -m ipykernel install --user --name genaiSpatialplan`
- update environment.yml: `conda env export --name genaiSpatialplan --file environment.yml`
- update environment: `conda env update --name genaiSpatialplan --file environment.yml`

## Data acquisition and processing
The data acquisition and processing scripts and notebooks are located in the [code/data_acquisition](./code/data_acquisition) folder. The final model input dataset is a zarr file containing multiple data sources combined and aligned to the same spatial grid. The data acquisition and processing pipeline consists of the following steps:
1. OpenStreetMap and building heights data acquisition and processing to an xarray dataset
2. Landsat and temperature data acquisition and processing to an xarray dataset
3. PlanetScope data acquisition and processing to an xarray dataset
4. Combining all datasets and clipping rural areas to create the final model input dataset

### Running the pipeline
There is a pipeline for the data acquisition and processing to acquire the model input dataset. The pipeline can be configured to choose different regions and different temperature settings using the [config.yml](./config.yml) file. The pipeline is designed to be run on an HPC cluster using SLURM job scheduling. If you don't have access to an HPC cluster, you can also run the individual scripts and notebooks standalone as described below.

To run the pipeline:

1. Please ensure all parameters are set before running the pipeline. To check which regions are available, open [data/ghsl/ghsl_data.parquet](./data/ghsl/ghsl_data.parquet) - e.g. using pandas or geopandas.
2. Create a new workspace under your username on the HPC cluster. You can do this by running `ws_allocate <name> <duration>` in the HPC shell, e.g. `ws_allocate master 30` to create a workspace named `<username>-master` for 30 days.
3. Make sure to download and convert the respective building height dataset like done and explained in the notebook [osm_to_xarray.ipynb](./code/data_acquisition/osm_to_xarray.ipynb). For Germany there already is a parquet file containing the building height data [here](https://www.dropbox.com/scl/fi/g1krcq2zj5wb6letsf65m/building_heights_germany.parquet?rlkey=a8pmpqtlu9wowttvfxgcb5rjp&st=twctw6j3&dl=0) that you can download and save to [data/che_etal/Germany_Hungary_Iceland](./data/che_etal/Germany_Hungary_Iceland) for the pipeline to work on all German regions.
4. Create the conda environment like indicated above and activate it: `conda activate genaiSpatialplan`.
5. Download the Corine Landcover dataset from https://land.copernicus.eu/en/products/corine-land-cover/clc2018 and save it to the [data/corine](./data/corine) folder. Unfortunately, this dataset cannot be downloaded automatically due to the required user agreement, so you have to do this step manually. You will have to create an account at EU Copernicus and agree to the terms of use. After downloading, unzip the dataset and rename it to `Corine_Landcover_<year>` (rename the folder with DATA, Legend etc. - not the .tif file).
6. Submit the pipeline to the HPC cluster using the [`submit_pipeline.py`](./code/data_acquisition/slurm/submit_pipeline.py) script: `python submit_pipeline.py`. This script will automatically create jobs for all pipeline steps. To check the status, run `squeue -u <username>` on the HPC cluster.

### Running the scripts standalone
1. Create the conda environment like indicated above and activate it: `conda activate genaiSpatialplan`
2. Run the osm_to_xarray script with the region as environment variable using: 
```powershell
$env:REGION = "your_region" # e.g. "Leipzig"
python ./code/data_acquisition/slurm/osm_to_xarray.py
```
3. Run the landsat_to_xarray script with the region as environment variable using: 
```powershell
$env:REGION = "your_region" 
python ./code/data_acquisition/slurm/landsat_to_xarray.py
```
4. Run the planetscope_to_xarray notebook
5. Download the Corine Landcover dataset from https://land.copernicus.eu/en/products/corine-land-cover/clc2018 and save it to the [data/corine](./data/corine) folder. Unfortunately, this dataset cannot be downloaded automatically due to the required user agreement, so you have to do this step manually. You will have to create an account at EU Copernicus and agree to the terms of use. After downloading, unzip the dataset and rename it to `Corine_Landcover_<year>` (rename the folder with DATA, Legend etc. - not the .tif file).
6. Lastly, use the combine_datasets notebook to combine all datasets to the final model input dataset.


## Model
The model is built as a latent diffusion model with a Gan-styled autoencoder and a UNet based diffusion model. The code for the model training and evaluation is located in the [code/model](./code/model) folder. For an overview of the model architecture, please refer to the master thesis document.

### Training and evaluation

For a Quick Start overview and detailed information on how to train and evaluate the model, please refer to the [QUICK_REFERENCE.md](./code/model/QUICK_REFERENCE.md) file in the model folder.



## Working with the HPC cluster
The HPC cluster of the University of Leipzig can be used to run the training and evaluation of the models. It is a SLURM based cluster with multiple nodes and GPUs available. The following instructions will help you to connect to the HPC cluster and run the training and evaluation scripts.

### connect to a hpc shell + useful commands
- ssh login using: `ssh <username>@login01.sc.uni-leipzig.de`
- check available modules: `module avail`
- load required modules: `module load <module_name>`

### Copy from and to HPC
- copy files from local to HPC: `scp <local_file_path> <username>@login01.sc.uni-leipzig.de:<remote_file_path>`
- copy files from HPC to local: `scp <username>@login01.sc.uni-leipzig.de:<remote_file_path> <local_file_path>`

Steps:
1. create a new workspace on the HPC cluster: `ws_allocate <name> <duration>`, e.g. `ws_allocate genai_spatial 30`
2. Set a reminder email before a workspace expires with `ws_send_ical <workspace~name> "<your~email>", e.g. `ws_send_ical <username>-genai_spatial "<your_email>@example.com"`
3. Convert dataset to zip
4. Copy the local dataset to the HPC cluster using `scp` command. e.g for model_input_dataset.zarr `scp model_input_dataset.zarr.zip <username>@login01.sc.uni-leipzig.de:/work/<username>-genai_spatial`
5. Unzip the dataset on the HPC cluster using `unzip model_input_dataset.zarr.zip`


### connect to a jupyter hub server on the HPC cluster
- select 'existing jupyter hub server' in VS Code in the kernel selection
- create a new session on the jupyter hub
- generate a token in the jupyter hub session
- set url to `https://lab.sc.uni-leipzig.de/jupyter/`
- set username to your hpc username
- set password/token to the generated token


## Special thanks
Special thanks to the following repositories and pages:
- https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main?tab=MIT-1-ov-file for building a great latent diffusion base. Several parts of this codebase are adapted from there.
- https://github.com/usuyama/pytorch-unet for a great starting point in understanding and implementing UNet architectures in PyTorch.
- https://gitlab.com/smart-quart/modulbaukasten for the learnings on working with OSM . (personally worked on this project as a student assistant)
- https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Frequently_used_code/Rasterise_vectorise.html for vectorizing rasters.