
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
    
    # get data for the i-th window:
    Extra = get_Extra(idx)
    model_data, initial_guesses, prior_params, response = load_data(idx)

    # setup model:
    m = CCNmodel_m1(Extra, model_data)

    # setup posterior:
    log_posterior = pints.LogPosterior(
        KnownSigmaGaussianLogLikelihood(m, response),
        joint_CauchyPrior(prior_params)
    )

    # intialize MCMC chains:
    initial_parameters = np.array(initial_guesses)
    x0 = [
        initial_parameters*0.98,
        initial_parameters*0.99,
        initial_parameters*1.01,
        initial_parameters*1.02,
        ]
    
    # setup optimisation controller:
    mcmc = pints.MCMCController(log_posterior, MCMC_SETTINGS['chains'], x0, method=pints.DreamMCMC)
    mcmc.set_initial_phase_iterations(MCMC_SETTINGS['burn_in'])
    mcmc.set_max_iterations(MCMC_SETTINGS['max_iterations'])
    mcmc.set_log_to_screen(False)

    # run MCMC:
    print(f"Running optimization for i={idx}")
    samples = mcmc.run()
    print(f"Done i={idx}")

    # save results:
    save_chain_results(samples, idx)