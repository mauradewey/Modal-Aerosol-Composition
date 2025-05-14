
# main.py

import os
from run_mcmc import run_mcmc_for_CCNwindow
import warnings

warnings.simplefilter('always')  # Log all warnings

# Get observation index from SLURM array task ID
#obs_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
obs_index = 20
# Run optimization
run_mcmc_for_CCNwindow(obs_index)