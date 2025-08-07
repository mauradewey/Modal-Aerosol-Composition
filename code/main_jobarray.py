
# main_jobarray.py

# This script runs the MCMC simulation for each CCN window in parallel.
# parallelization is done with SLURM job arrays.

from run_mcmc import run_mcmc_for_CCNwindow
import warnings
import argparse

warnings.simplefilter('always')  # Log all warnings

def main():
    parser = argparse.ArgumentParser(description="Run MCMC for a specific CCN window.")
    parser.add_argument('--idx', type=int, required=True, help='Index of the CCN window (e.g., SLURM_ARRAY_TASK_ID)')
    args = parser.parse_args()

    idx = args.idx - 1
    print(f"Starting MCMC for index: {idx}")
    
    try:
        run_mcmc_for_CCNwindow(idx)
        print(f"[INFO] Finished MCMC for CCN window index: {idx}")
    except Exception as e:
        print(f"[ERROR] Failed at index {idx} with error: {e}")

if __name__ == "__main__":
    main()
