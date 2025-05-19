#!/bin/bash
#
#SBATCH -A naiss2024-1-3
#SBATCH -J CCN_MCMC
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mail-user=maura.dewey@misu.su.se  
#SBATCH --mail-type=FAIL
#SBATCH -o logs/slurm-dask-mcmc_%j.out
#SBATCH -e logs/slurm-dask-mcmc_%j.out
           
module load Miniforge/24.7.1-2-hpc1
conda activate mcmc_env

python code/main_dask.py