
# main_dask.py

# This script initializes a Dask client and runs the MCMC simulation for each CCN window in parallel.
# The MCMC setting, model, likelihood, and prior classes are defined in the run_mcmc.py file.

from dask import delayed, compute
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from code.run_mcmc_m4 import run_mcmc_for_CCNwindow
import warnings
import pickle

warnings.simplefilter('always')  # Log all warnings

def main():
    cluster = SLURMCluster(       
        account='naiss2025-1-5',      
        cores=32,    
        processes=32,
        memory="96GiB",             
        walltime='06:00:00',
        job_script_prologue=[
        'module load Miniforge/24.7.1-2-hpc1',
        'conda activate mcmc_env',
        'export PYTHONPATH=/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/code:$PYTHONPATH'
    ] 
    )

    cluster.scale(jobs=12)  # adjust to number of nodes you want

    # Connect Dask client
    client = Client(cluster)
    print("Dask client info: ", client)


    # Get CCN windows to run:
    with open('/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/results/missing_windows_summary_m4_org_and_sizeparams.pickle', 'rb') as f:
        missing_windows = pickle.load(f)

    tasks = [delayed(run_mcmc_for_CCNwindow)(i) for i in missing_windows]

    # Compute the results in parallel
    compute(*tasks, scheduler='distributed')
    client.close()
    print("All MCMC runs completed.")


if __name__ == "__main__":
    main()
