# Repository accompanying the master thesis on generative AI in spatial planning

| title: A heat island aware generative urban layout model. Can artificial intelligence generate new scenarios of city areas optimized for future climates, for more informed decision making?

| author: Jannes Kruse


## Installation

**First time installation:**
```bash
conda env create -f environment.yml
conda activate genaiSpatialplan
pip install -e .  # Installs project in editable mode so modules can imported from everywhere
python -m ipykernel install --user --name genaiSpatialplan
```

Then on every project session, activate the environment and install any missing packages with a single command:

**Linux/Mac/SLURM:**
```bash
source ./activate.sh   # Updates environment + installs package + activates
```

**Windows:**
```powershell
.\activate.ps1
```


**Update environment when dependencies change:**
```bash
conda env update -f environment.yml --prune
```

**Update environment.yml (after installing new packages):**
```bash
conda env export --name genaiSpatialplan --file environment.yml
```

## Data acquisition and processing
The data acquisition and processing scripts and notebooks are located in the [code/data_acquisition](./code/data_acquisition) folder. The final model input dataset is a zarr file containing multiple data sources combined and aligned to the same spatial grid. The data acquisition and processing pipeline consists of the following steps:
1. **OpenStreetMap** and building heights data acquisition and rasterization — `osm_to_xarray.py`
2. **Landsat** land-surface temperature and DWD weather station data — `landsat_to_xarray.py`
3. **PlanetScope** high-resolution satellite imagery — two-step process:
   - `request_planetscope.py` searches the Planet API and downloads scene assets
   - `process_planetscope.py` processes all dates in parallel using multiprocessing and combines them into a single Zarr cube
4. **Combining** all datasets and clipping rural areas — `combine_datasets.py`

### Pipeline architecture

The pipeline orchestrator [`submit_pipeline.py`](./code/data_acquisition/tools/submit_pipeline.py) submits SLURM jobs per region with the following dependencies:
- **PlanetScope depends on Landsat** — `request_planetscope` uses the Landsat Zarr's time dimension to know which dates to request.
- **Combine depends on all three upstream steps** — it is automatically submitted when the last of `osm_to_xarray` or `process_planetscope` finishes, or on the next run of `submit_pipeline.py`

### Job tracking

Each script records its status to a per-region CSV file at `{big_data_storage_path}/processed/{region}/jobs/{script_name}.csv`. Each run appends a row with:

| Column | Description |
|--------|-------------|
| `job_id` | SLURM job ID (or `"local"` when running outside SLURM) |
| `script_name` | Logical script name |
| `start_time` | ISO timestamp when the job started |
| `end_time` | ISO timestamp when the job finished |
| `duration_seconds` | Wall-clock duration |
| `status` | `in_progress`, `completed`, or `failed` |
| `error_message` | Error details (if failed) |

`submit_pipeline.py` reads these CSVs to decide what to submit:
- **`None`** (no CSV) or **`failed`** → submit the job
- **`in_progress`** or **`completed`** → skip

This means you can safely re-run `submit_pipeline.py` after a failure — it will only resubmit failed steps, not duplicate running or completed jobs.

### Running the pipeline
The pipeline can be configured to choose different regions and different temperature settings using the [code/data_acquisition/config.yml](./code/data_acquisition/config.yml) file. The pipeline is designed to be run on an HPC cluster using SLURM job scheduling. If you don't have access to an HPC cluster, you can also run the individual scripts standalone as described below.

To run the pipeline:

1. Please ensure all parameters are set before running the pipeline. To check which regions are available, open [data/ghsl/ghsl_data.parquet](./data/ghsl/ghsl_data.parquet) - e.g. using pandas or geopandas.
2. Create a new workspace under your username on the HPC cluster. You can do this by running `ws_allocate <name> <duration>` in the HPC shell, e.g. `ws_allocate master 30` to create a workspace named `<username>-master` for 30 days. Then update the config file [code/data_acquisition/config.yml](./code/data_acquisition/config.yml) for the new storage path (`big_data_storage_path` parameter)
3. Make sure to download and convert the respective building height dataset like done and explained in the notebook [osm_to_xarray.ipynb](./code/data_acquisition/osm_to_xarray.ipynb). For Germany there already is a parquet file containing the building height data [here](https://www.dropbox.com/scl/fi/g1krcq2zj5wb6letsf65m/building_heights_germany.parquet?rlkey=a8pmpqtlu9wowttvfxgcb5rjp&st=twctw6j3&dl=0) that you can download and save to [data/che_etal/Germany_Hungary_Iceland](./data/che_etal/Germany_Hungary_Iceland) for the pipeline to work on all German regions.
4. Create the conda environment like indicated above and activate it: `conda activate genaiSpatialplan`.
5. Download the Corine Landcover dataset from https://land.copernicus.eu/en/products/corine-land-cover/clc2018 and save it to the [data/corine](./data/corine) folder. Unfortunately, this dataset cannot be downloaded automatically due to the required user agreement, so you have to do this step manually. You will have to create an account at EU Copernicus and agree to the terms of use. After downloading, unzip the dataset and rename it to `Corine_Landcover_<year>` (rename the folder with DATA, Legend etc. - not the .tif file).
6. Navigate to the SLURM scripts directory and submit the pipeline:
```bash
cd code/data_acquisition/tools/slurm
python ../submit_pipeline.py
```
This will automatically submit jobs for all regions and pipeline steps. To check status, run `squeue -u <username>` on the HPC cluster. You can re-run `submit_pipeline.py` at any time — it will only resubmit failed or not-yet-started steps.

### Running the scripts standalone
1. Create the conda environment like indicated above and activate it: `conda activate genaiSpatialplan`
2. Navigate to the tools directory: `cd code/data_acquisition/tools`
3. Run the OSM script:
```bash
python osm_to_xarray.py --REGION Leipzig
```
4. Run the Landsat script:
```bash
python landsat_to_xarray.py --REGION Leipzig
```
5. Run the PlanetScope request script (requires a completed Landsat Zarr):
```bash
python request_planetscope.py --REGION Leipzig --LANDSAT_ZARR_NAME <path_to_landsat.zarr> --REGION_FILENAMES_JSON '<json_string>'
```
6. Run the PlanetScope processing script (requires the completed request script):
```bash
python process_planetscope.py --REGION Leipzig --LANDSAT_ZARR_NAME <path_to_landsat.zarr> --FILENAMES "file1.parquet:file2.parquet:..."
```
The filenames should be printed in the right format with ':' seperation at the end of step 5 (the run command). 
7. Download the Corine Landcover dataset from https://land.copernicus.eu/en/products/corine-land-cover/clc2018 and save it to the [data/corine](./data/corine) folder. Unfortunately, this dataset cannot be downloaded automatically due to the required user agreement, so you have to do this step manually. You will have to create an account at EU Copernicus and agree to the terms of use. After downloading, unzip the dataset and rename it to `Corine_Landcover_<year>` (rename the folder with DATA, Legend etc. - not the .tif file).
8. Combine all datasets into the final model input dataset:
```bash
python combine_datasets.py --REGION Leipzig
```

| Note, that also here, you can tweak the settings in the [code/data_acquisition/config.yml](./code/data_acquisition/config.yml) file before running the scripts.


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
- request interactive shell for running scripts: `srun --pty -p paula --gpus=a30:4 --ntasks=1 --cpus-per-task=8 --mem=64G --time=01:00:00 bash`

e.g. to run the diffusion script with 4 GPUs:

```bash
# 1. Request interactive session (ntasks=1 for bash, but reserve 4 GPUs)
srun --pty -p paula --gpus=a30:4 --ntasks=1 --cpus-per-task=8 --mem=64G --time=02:00:00 bash

# 2. Activate environment
conda activate genaiSpatialplan
cd ~/masterthesis_genai_spatialplan/code/model/tools

# 3. Set up distributed environment variables
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export OMP_NUM_THREADS=2  # CPUs per GPU (8 CPUs / 4 GPUs)

# 4. Launch with torchrun (spawns 4 processes, one per GPU)
# without checkpoint
torchrun --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_diffusion_inpainting_ddp.py --config two_stage_12.yml --mode semantic

# with checkpoint
torchrun --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_diffusion_inpainting_ddp.py --config two_stage_12.yml --mode semantic --load_checkpoint semantic_diffusion_ckpt.pth

```

> **Note:** Use `torchrun` to spawn DDP processes. Nested `srun` can cause GRES specification conflicts.

> **Note* If you want to reattach to a running job you can run `srun --pty --overlap --jobid <your-jobid> bash`


### Copy from and to HPC
- copy files from local to HPC: `scp <local_file_path> <username>@login01.sc.uni-leipzig.de:<remote_file_path>`
- copy files from HPC to local: `scp <username>@login01.sc.uni-leipzig.de:<remote_file_path> <local_file_path>`

Steps:
1. create a new workspace on the HPC cluster: `ws_allocate <name> <duration>`, e.g. `ws_allocate genai_spatial 30`
2. Set a reminder email before a workspace expires with `ws_send_ical <workspace~name> "<your~email>", e.g. `ws_send_ical <username>-genai_spatial "<your_email>@example.com"`
3. Either: 
- Convert dataset to zip
- Copy the local dataset to the HPC cluster using `scp` command. e.g for model_input_dataset.zarr `scp model_input_dataset.zarr.zip <username>@login01.sc.uni-leipzig.de:/work/<username>-genai_spatial`
- Unzip the dataset on the HPC cluster using `unzip model_input_dataset.zarr.zip`
4. Or copy recursively using `scp -r model_input_dataset.zarr <username>@login01.sc.uni-leipzig.de:/work/<username>-genai_spatial`


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