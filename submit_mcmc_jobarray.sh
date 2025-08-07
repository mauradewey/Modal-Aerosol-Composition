#!/bin/bash
#
#SBATCH -A naiss2024-1-3
#SBATCH -J CCN_MCMC
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mail-user=maura.dewey@misu.su.se  
#SBATCH --mail-type=END,FAIL
#SBATCH -o logs/slurm-mcmc_%j.out
#SBATCH -e logs/slurm-mcmc_%j.out
#SBATCH --array=1-10:1%1000
           
module load Miniforge/24.7.1-2-hpc1
conda activate mcmc_env

python code/main_jobarray.py --idx $SLURM_ARRAY_TASK_ID