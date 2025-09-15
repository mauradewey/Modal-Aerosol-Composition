
# run_mcmc_m3.py

# Optimize Organic density and Kappa only.

# This file runs the DREAM Monte Carlo Markov Chain (MCMC) simulation for CCN closure.
# Here we choose the specific model, likelihood, and prior classes.
# Make sure that MCMC settings are set in the config.py file.

from models import CCNmodel_m3
from likelihoods import KnownSigmaGaussianLogLikelihood
#from priors import joint_CauchyPrior
from config import get_Extra, save_chain_results, get_initial_guesses_near_base
import pints
import numpy as np
import pdb
import pandas as pd

def run_mcmc_for_CCNwindow(idx):

    base_fname = '20k_m3_orgs_only'  # Base filename for saving MCMC results

    #restart_dir = 'm2_40k_logparams' #folder with existing chains to restart from

    MCMC_SETTINGS = {
    'max_iterations': 20000,  # Maximum number of MCMC iterations
    'burn_in': 10000,     # Number of initial phase iterations
    'chains': 5,         # Number of MCMC chains
    'restart': False,  # Whether to restart from existing chains
    }

    try:
        
        # get data for the i-th window:
        Extra = get_Extra(idx)

        NSD1 = np.array(pd.read_csv('/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/input_data/NSD_mode1.csv').iloc[idx,1:].values.tolist())  # Load NSD data for mode 1
        NSD2 = np.array(pd.read_csv('/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/input_data/NSD_mode2.csv').iloc[idx,1:].values.tolist())  # Load NSD data for mode 2
        response = pd.read_csv('/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/input_data/CCN.csv', header=None, skiprows=idx+1, nrows=1).drop(columns=0).values[0]
        model_data_NSD = (NSD1, NSD2)

        # setup model:
        m = CCNmodel_m3(Extra, model_data_NSD)

        # setup priors:
        prior = pints.UniformLogPrior([1000, 0.05], [3000, 0.15]) # Density between 1000 and 3000 kg/m3 and Kappa between 0.05 and 0.15

        # setup posterior:
        log_posterior = pints.LogPosterior(
            KnownSigmaGaussianLogLikelihood(m, response),
            prior 
        )

        # get starting parameter values:
        initial_guesses = np.array([1500, 0.12])  # Initial guess for Density and Kappa
        x0 = get_initial_guesses_near_base(idx, log_posterior, prior, np.array(initial_guesses), n_chains=MCMC_SETTINGS['chains'])


        # setup MCMC controller:
        mcmc = pints.MCMCController(log_posterior, MCMC_SETTINGS['chains'], x0, method=pints.DreamMCMC)
        mcmc.set_initial_phase_iterations(MCMC_SETTINGS['burn_in'])
        mcmc.set_max_iterations(MCMC_SETTINGS['max_iterations'])
        mcmc.set_log_to_screen(False)
        mcmc.sampler().set_nCR(5)

        # run MCMC:
        samples = mcmc.run()
        print(f"Done MCMC for window {idx}")

        # save chains:
        print(f"Saving chains for window {idx}.")
        save_chain_results(base_fname, samples, MCMC_SETTINGS['chains'], idx)


    except Exception as e:
        print(f"Error in MCMC for CCN window {idx}: {e}")
        import traceback
        traceback.print_exc()
        return f'Failed for window {idx}'


