#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --job-name="Combine_Datasets"
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de #please indicate your own email here
#SBATCH --mail-type=ALL
#SBATCH -o "outputs/combine_datasets.%j.txt" #j for the job id

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Run the final combination script
python3 -u combine_datasets.py

done

    

