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

'''
Loop though MCMC chains and summarize the posterior distributions of parameters.
Example usage in terminal:
python summarize_mcmc_posteriors.py --chain_folder m2_30k_5chains --output_file summary_30k_m2_5chains.csv --burn_in 15000
'''

def main(chain_folder, output_file, burn_in):

    bimodal_params = pd.read_csv('/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/input_data/bimodal_params_windows.csv') 
    mcmc_params = pd.DataFrame({'datetime': bimodal_params['datetime']})
    missing_windows = []

    # Loop over parameter windows
    for ii in range(len(mcmc_params)):
        try:
            files = sorted(glob.glob(f'/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/chains/{chain_folder}/*_{str(ii)}_*.csv'))

            M_org1_chains = pints.io.load_samples(files[0])
            D1_chains     = pints.io.load_samples(files[1])
            N1_chains     = pints.io.load_samples(files[2])
            D2_chains     = pints.io.load_samples(files[3])
            N2_chains     = pints.io.load_samples(files[4])

            #burn_in = 15000
            burn_in_ratio = np.round(burn_in / M_org1_chains.shape[1], 2)

            param_dict = {
                'M_org1': M_org1_chains,
                'D1': D1_chains,
                'N1': N1_chains,
                'D2': D2_chains,
                'N2': N2_chains,
            }

            for param, chains in param_dict.items():
                samples = chains[:, burn_in:]
                mcmc_params.at[ii, f'{param}_mean']   = np.mean(samples)
                mcmc_params.at[ii, f'{param}_median'] = np.median(samples)
                mcmc_params.at[ii, f'{param}_mode']    = mode(samples,axis=None)[0]
                mcmc_params.at[ii, f'{param}_std']    = np.std(samples)
                mcmc_params.at[ii, f'{param}_rhat']   = pints.rhat(chains, burn_in_ratio)
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
    output_path = f'/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/results/{output_file}'
    mcmc_params.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_file}")
    #if missing_windows:
        #print(f"Missing windows: {missing_windows}")
    #    missing_windows_df = pd.DataFrame({'missing_windows': missing_windows})
    #    missing_windows_df.to_csv(f'/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/results/missing_windows_{output_file}', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summarize MCMC chains into parameter statistics.")
    parser.add_argument('--chain_folder', type=str, required=True, help='folder where chains are stored for a particular experiment.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV.')
    parser.add_argument('--burn_in', type=int, required=True, help='Number of samples to discard as burn-in.')
    args = parser.parse_args()

    main(args.chain_folder, args.output_file, args.burn_in)
