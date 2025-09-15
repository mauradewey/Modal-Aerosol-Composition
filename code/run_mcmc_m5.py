# run_mcmc_m5.py

# Optimize BC mass fraction, Organic mass fraction, and size distribution parameters.

# This file runs the DREAM Monte Carlo Markov Chain (MCMC) simulation for CCN closure.
# Here we choose the specific model, likelihood, and prior classes.
# Make sure that MCMC settings are set in the config.py file.

from models import CCNmodel_m5
from likelihoods import KnownSigmaGaussianLogLikelihood
from priors import joint_CauchyPrior
from config import get_Extra, save_chain_results, get_initial_guesses_near_base, load_data
import pints
import numpy as np
import pandas as pd

def run_mcmc_for_CCNwindow(idx):

    base_fname = '40k_m5_bc_morg'  # Base filename for saving MCMC results

    #restart_dir = 'm2_40k_logparams' #folder with existing chains to restart from

    MCMC_SETTINGS = {
    'max_iterations': 40000,  # Maximum number of MCMC iterations
    'burn_in': 20000,     # Number of initial phase iterations
    'chains': 5,         # Number of MCMC chains
    'restart': False,  # Whether to restart from existing chains
    }

    try:
        
        # get data for the i-th window:
        Extra = get_Extra(idx)
        model_data, initial_guesses, prior_params, response = load_data(idx)
        M_BC1_initial_guess = pd.read_csv('/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/input_data/M_BC1_initial_guess.txt', header=None).iloc[idx,0]

        # setup model:
        m = CCNmodel_m5(Extra, model_data)

        # setup priors:
        org_prior = joint_CauchyPrior(prior_params, initial_guesses[0])  # M_org1_initial_guess is the first element of initial_guesses
        bc_prior = pints.HalfCauchyLogPrior(M_BC1_initial_guess, 0.5)  # Half-Cauchy prior for M_BC1
        prior = pints.ComposedLogPrior(bc_prior, org_prior)

        # setup posterior:
        log_posterior = pints.LogPosterior(
            KnownSigmaGaussianLogLikelihood(m, response),
            prior 
        )


        initial_guesses_all = np.concatenate(([M_BC1_initial_guess], initial_guesses)) # Initial guess for Density and Kappa
        x0 = get_initial_guesses_near_base(idx, log_posterior, prior, np.array(initial_guesses_all), n_chains=MCMC_SETTINGS['chains'])

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
        save_chain_results(base_fname, samples, MCMC_SETTINGS['chains'], idx)


    except Exception as e:
        print(f"Error in MCMC for CCN window {idx}: {e}")
        import traceback
        traceback.print_exc()
        return f'Failed for window {idx}'


