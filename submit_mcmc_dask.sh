#!/bin/bash
#
#SBATCH -A naiss2025-1-5
#SBATCH -J CCN_MCMC
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mail-user=maura.dewey@misu.su.se  
#SBATCH --mail-type=FAIL
#SBATCH -o logs/slurm-dask-mcmc_%j.out
#SBATCH -e logs/slurm-dask-mcmc_%j.out
           
module load Miniforge/24.7.1-2-hpc1
conda activate mcmc_env

python code/main_dask.py

#python code/summarize_mcmc_posteriors.py --chain_folder m4_org_and_sizeparams --output_file summary_m4_org_and_sizeparams --sample_len 20000 --model_v m4