
# run_mcmc.py

# This file runs the DREAM Monte Carlo Markov Chain (MCMC) simulation for CCN closure.
# Here we choose the specific model, likelihood, and prior classes.
# Make sure that MCMC settings are set in the config.py file.

from models import CCNmodel_m3
from likelihoods import KnownSigmaGaussianLogLikelihood
#from priors import joint_CauchyPrior
from config import get_Extra, save_chain_results, MCMC_SETTINGS
import pints
import numpy as np
import pdb
import pandas as pd

def run_mcmc_for_CCNwindow(idx):

    try:
        
        # get data for the i-th window:
        Extra = get_Extra(idx)
        #model_data, initial_guesses, prior_params, response = load_data(idx)
        NSD1 = np.array(pd.read_csv('../input_data/NSD_mode1.csv').iloc[idx,1:].values.tolist())  # Load NSD data for mode 1
        NSD2 = np.array(pd.read_csv('../input_data/NSD_mode2.csv').iloc[idx,1:].values.tolist())  # Load NSD data for mode 2
        response = pd.read_csv('../input_data/CCN.csv', header=None, skiprows=idx+1, nrows=1).drop(columns=0).values[0]
        model_data = (NSD1, NSD2)

        # setup model:
        m = CCNmodel_m3(Extra, model_data)

        # setup priors:
        #prior = joint_CauchyPrior(prior_params, initial_guesses[0])  # M_org1_initial_guess is the first element of initial_guesses
        prior = pints.UniformLogPrior([0.05, 1000], [0.15, 3000]) # Kappa between 0.05 and 0.15 and Density between 1000 and 3000 kg/m3

        # setup posterior:
        log_posterior = pints.LogPosterior(
            KnownSigmaGaussianLogLikelihood(m, response),
            prior 
        )

        # get starting parameter values:
        #if MCMC_SETTINGS['restart']:
        #    print(f"Restarting MCMC for CCN window {idx} from existing chains...")
        #    x0 = get_restart_samples(idx, MCMC_SETTINGS['chains'])
            
        #if MCMC_SETTINGS['restart'] == False:
        #    print(f"Running MCMC for CCN window {idx}...")
        #    x0 = get_initial_guesses_near_base(idx, log_posterior, prior, np.array(initial_guesses), n_chains=MCMC_SETTINGS['chains'])

        x0 = np.array([0.12, 1500])  # Initial guess for Kappa and Density
 

        # setup parameter transformation:
        #transform = pints.LogTransformation(n_parameters=5)

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
        save_chain_results(samples, MCMC_SETTINGS['chains'], idx)


    except Exception as e:
        print(f"Error in MCMC for CCN window {idx}: {e}")
        import traceback
        traceback.print_exc()
        return f'Failed for window {idx}'


