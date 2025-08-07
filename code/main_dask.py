
# main_dask.py

# This script initializes a Dask client and runs the MCMC simulation for each CCN window in parallel.
# The MCMC setting, model, likelihood, and prior classes are defined in the run_mcmc.py file.

from dask import delayed, compute
from dask.distributed import Client
from run_mcmc import run_mcmc_for_CCNwindow
import warnings
import pickle

warnings.simplefilter('always')  # Log all warnings

def main():
    # Initialize Dask client
    client = Client(n_workers=32, threads_per_worker=1)
    print('Initialized Dask cluster with 32 workers.')

    # Number of CCN windows:
    #num_windows = 3000
    with open('/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/results/not_convergedbelow3_m2_40k_logparams.pickle', 'rb') as f:
        missing_windows = pickle.load(f)
   
    tasks = [delayed(run_mcmc_for_CCNwindow)(i) for i in missing_windows[400:600]] 

    # Compute the results in parallel
    compute(*tasks, scheduler='distributed')
    client.close()
    print("All MCMC runs completed.")


if __name__ == "__main__":
    main()
