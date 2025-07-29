import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import mode
import pints
import pints.io
import argparse
import pdb
import pickle

'''
Loop though MCMC chains and summarize the posterior distributions of parameters.
Example usage in terminal:
python summarize_mcmc_posteriors.py --chain_folder m2_30k_5chains --output_file summary_30k_m2_5chains --sample_len 10000

sample_len: number of samples at the end of the chains to use for statistics.
'''

def main(chain_folder, output_file, sample_len):

    bimodal_params = pd.read_csv('/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/input_data/bimodal_params_windows.csv') 
    mcmc_params = pd.DataFrame({'datetime': bimodal_params['datetime']})
    missing_windows = []

    # Loop over CCN obs windows:
    for ii in range(len(mcmc_params)):
        try:

            # 1. Get chains for this window:
            files = sorted(glob.glob(f'/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/chains/{chain_folder}/mcmc*_{str(ii)}_*.csv'))

            # if there are more than 5 files, we assume these are restarts and we use the restart files:
            if len(files) > 5: #use restarts
                files = sorted(glob.glob(f'/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/chains/{chain_folder}/mcmc*restarts*_{str(ii)}_*.csv'))

            # 2. Load chains:
            M_org1_chains = pints.io.load_samples(files[0])
            D1_chains     = pints.io.load_samples(files[1])
            N1_chains     = pints.io.load_samples(files[2])
            D2_chains     = pints.io.load_samples(files[3])
            N2_chains     = pints.io.load_samples(files[4])

            # 3. Check R-hat convergence:          
            rhat_cutoff = sample_len/len(M_org1_chains[0]) # calculate rhat cutoff based on how many iterations we want to use for statistics (usually 0.5)
            all_chains = np.stack([M_org1_chains, D1_chains, N1_chains, D2_chains, N2_chains], axis=2)

            # if all chains have converged, use them all for statistics:
            if all(pints.rhat(all_chains, rhat_cutoff)<1.5):
                good_chains_idx = list(range(all_chains.shape[0]))

            # else, we need to check for outliers:
            else:
                # detect outlier chains in the last sample_len samples:
                bad_chains = detect_outlier_chains(all_chains[:, -sample_len:, :], z_thresh=1.5)

                # discard bad chains if there is only one or two:
                if len(bad_chains) in (1,2):
                    good_chains_idx = [i for i in range(all_chains.shape[0]) if i not in bad_chains]
                
                # else, there are more than two (maybe bimodal, etc), so we keep all chains:
                else:
                    good_chains_idx = list(range(all_chains.shape[0]))
                
            # 4. Calculate and store posterior statistics using all good chains:
            param_dict = {
                'M_org1': M_org1_chains,
                'D1': D1_chains,
                'N1': N1_chains,
                'D2': D2_chains,
                'N2': N2_chains,
            }

            for param, chains in param_dict.items():
                rhat_val = pints.rhat(chains[good_chains_idx,:], rhat_cutoff)
                samples = chains[good_chains_idx, -sample_len:] #use last sample_len samples in good chains to calculate statistics

                mcmc_params.at[ii, f'{param}_mean']   = np.mean(samples)
                mcmc_params.at[ii, f'{param}_median'] = np.median(samples)
                mcmc_params.at[ii, f'{param}_mode']    = mode(samples,axis=None)[0]
                mcmc_params.at[ii, f'{param}_std']    = np.std(samples)
                mcmc_params.at[ii, f'{param}_rhat']   = rhat_val
                mcmc_params.at[ii, f'{param}_25']     = np.percentile(samples, 2.5)
                mcmc_params.at[ii, f'{param}_975']    = np.percentile(samples, 97.5)
                mcmc_params.at[ii, f'{param}_skew']   = skew(samples, axis=None)


        except Exception as e:
            print(f"Missing or failed chains for window {ii}: {e}")
            all_cols_to_nan = mcmc_params.columns.difference(['datetime'])
            mcmc_params.loc[ii, all_cols_to_nan] = np.nan
            missing_windows.append(ii)
            continue

    # Save the final summary
    output_path = f'/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/results/{output_file}.csv'
    mcmc_params.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_file}.csv")

    # Save missing windows
    if missing_windows:
        with open(f'/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/results/missing_windows_{output_file}.pickle', 'wb') as f:
            pickle.dump(missing_windows, f, pickle.HIGHEST_PROTOCOL)

def detect_outlier_chains(samples, z_thresh=1.5):
    """
    samples: np.ndarray of shape (n_chains, n_samples, n_params)
    Returns list of chain indices likely responsible for bad R-hat
    """
    n_chains, n_samples, n_params = samples.shape
    bad_chains = set()

    for p in range(n_params):
        param_chains = samples[:, :, p]
        param_means = param_chains.mean(axis=1)
        mean = np.mean(param_means)
        std = np.std(param_means)

        z_scores = np.abs((param_means - mean) / std)
        for i, z in enumerate(z_scores):
            if z > z_thresh:
                bad_chains.add(i)

    return list(bad_chains)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summarize MCMC chains into parameter statistics.")
    parser.add_argument('--chain_folder', type=str, required=True, help='folder where chains are stored for a particular experiment.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV.')
    parser.add_argument('--sample_len', type=int, required=True, help='Number of samples at end of chains to use for stats.')
    args = parser.parse_args()

    main(args.chain_folder, args.output_file, args.sample_len)
