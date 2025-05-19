
# run_mcmc.py

# This file runs the DREAM Monte Carlo Markov Chain (MCMC) simulation for CCN closure.
# Here we choose the specific model, likelihood, and prior classes.
# The MCMC settings are defined in config.py.

from models import CCNmodel_m1
from likelihoods import KnownSigmaGaussianLogLikelihood
from priors import joint_CauchyPrior
from config import MCMC_SETTINGS, get_Extra, load_data, save_chain_results
import pints
import numpy as np

def run_mcmc_for_CCNwindow(idx):

    try:
        print(f"Running MCMC for CCN window {idx}...")
    
        # get data for the i-th window:
        Extra = get_Extra(idx)
        model_data, initial_guesses, prior_params, response = load_data(idx)

        # setup model:
        m = CCNmodel_m1(Extra, model_data)
        prior = joint_CauchyPrior(prior_params)

        # setup posterior:
        log_posterior = pints.LogPosterior(
            KnownSigmaGaussianLogLikelihood(m, response),
            prior
        )

        # intialize MCMC chains:
        #initial_parameters = np.array(initial_guesses)
        #x0 = [
        #    initial_parameters*0.98,
        #    initial_parameters*0.99,
        #    initial_parameters*1.01,
        #    initial_parameters*1.02,
        #    ]
        x0 = get_initial_guesses(idx, log_posterior, prior)
        
        # setup optimisation controller:
        mcmc = pints.MCMCController(log_posterior, MCMC_SETTINGS['chains'], x0, method=pints.DreamMCMC)
        mcmc.set_initial_phase_iterations(MCMC_SETTINGS['burn_in'])
        mcmc.set_max_iterations(MCMC_SETTINGS['max_iterations'])
        mcmc.set_log_to_screen(False)

        # run MCMC:
        samples = mcmc.run()
        print(f"Done for window {idx}")

        # save results:
        save_chain_results(samples, idx)
        
        print(f"Finished MCMC for CCN window {idx}.")

    except Exception as e:
        print(f"Error in MCMC for CCN window {idx}: {e}")
        import traceback
        traceback.print_exc()
        return f'Failed for window {idx}'


def get_initial_guesses(idx, posterior, prior, n_chains=MCMC_SETTINGS['chains'], max_attempts=1000):
    """
    Generate initial guesses for the MCMC run from a given prior
    and check that it gives a valid posterior.
    """
    x0 = []
    attempts = 0

    while len(x0) < n_chains and attempts < max_attempts:
        sample = prior.sample().flatten()
        if np.isfinite(posterior(sample)):
            x0.append(sample)
        attempts += 1
    if attempts == max_attempts:
        raise ValueError(f"Could not generate {n_chains} valid initial guesses from the prior for window {idx}.")
    return x0