
# main.py

# This script initializes a Dask client and runs the MCMC simulation for each CCN window in parallel.

from dask import delayed, compute
from dask.distributed import Client
from run_mcmc import run_mcmc_for_CCNwindow
import warnings

warnings.simplefilter('always')  # Log all warnings

def main():
    # Initialize Dask client
    client = Client(n_workers=32, threads_per_worker=1)
    print('Initialized Dask cluster with 32 workers.')

    # Number of CCN windows:
    num_windows = 100

    tasks = [delayed(run_mcmc_for_CCNwindow)(i) for i in range(num_windows)]

    # Compute the results in parallel
    compute(*tasks, scheduler='distributed')
    print("All dask tasks completed.")

if __name__ == "__main__":
    main()
