#!/bin/bash
#
#SBATCH -A naiss2025-1-5
#SBATCH -J CCN_MCMC
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mail-user=maura.dewey@misu.su.se  
#SBATCH --mail-type=FAIL
#SBATCH -o logs/slurm-dask-mcmc_%j.out
#SBATCH -e logs/slurm-dask-mcmc_%j.out
           
module load Miniforge/24.7.1-2-hpc1
conda activate mcmc_env

python code/main_dask_3.py

#python code/summarize_mcmc_posteriors.py --chain_folder m2_40k_logparams --output_file summary_40k_m2_logparams_v2 --sample_len 20000