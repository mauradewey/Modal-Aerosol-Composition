#!/bin/bash
#
#SBATCH -A naiss2024-1-3
#SBATCH -J CCN_MCMC
#SBATCH -t 04:00:00
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH --mail-user=maura.dewey@misu.su.se  
#SBATCH --mail-type=FAIL
#SBATCH -o logs/slurm-dask-mcmc_%j.out
#SBATCH -e logs/slurm-dask-mcmc_%j.out
#SBATCH --array=1-100:1%1000
           
module load Miniforge/24.7.1-2-hpc1
conda activate mcmc_env

python code/main_jobarray.py --idx $SLURM_ARRAY_TASK_ID