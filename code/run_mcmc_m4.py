
# run_mcmc_m4.py

# Optimize Organic density and Kappa, and size distribution parameters.

# This file runs the DREAM Monte Carlo Markov Chain (MCMC) simulation for CCN closure.
# Here we choose the specific model, likelihood, and prior classes.

from models import CCNmodel_m4
from likelihoods import KnownSigmaGaussianLogLikelihood
from config import get_Extra, save_chain_results, get_initial_guesses_near_base, load_data
import pints
import numpy as np


def run_mcmc_for_CCNwindow(idx):

    base_fname = '40k_m4_org_sizeparams'  # Base filename for saving MCMC results

    MCMC_SETTINGS = {
    'max_iterations': 40000,  # Maximum number of MCMC iterations
    'burn_in': 15000,     # Number of initial phase iterations
    'chains': 5,         # Number of MCMC chains
    'restart': False,  # Whether to restart from existing chains
    }

    try:
        
        # get data for the i-th window:
        Extra = get_Extra(idx)
        model_data, initial_guesses, prior_params, response = load_data(idx)

        # setup model:
        m = CCNmodel_m4(Extra, model_data)

        # setup priors:
        medians = np.round(prior_params['medians'],4)
        mad = np.round(prior_params['mad'],4)

        # Create a Cauchy log prior for each parameter which is truncated to be positive:
        size_priors = [pints.TruncatedGaussianLogPrior(medians[0], min(1, mad[0]), 0, 1000),    # diameters restricted to between 0 and 1000nm
                       pints.HalfCauchyLogPrior(medians[1], min(1, mad[1])),
                       pints.TruncatedGaussianLogPrior(medians[2], min(1, mad[2]), 0, 1000),
                       pints.HalfCauchyLogPrior(medians[3], min(1, mad[3]))]

        # Create Uniform priors for the organic properties:
        org_priors = pints.UniformLogPrior([1000, 0.05], [3000, 0.15]) # Density between 1000 and 3000 kg/m3 and Kappa between 0.05 and 0.15
        prior = pints.ComposedLogPrior(org_priors, *size_priors)

        # setup posterior:
        log_posterior = pints.LogPosterior(
            KnownSigmaGaussianLogLikelihood(m, response),
            prior 
        )


        initial_guesses_all = np.concatenate(([1500, 0.12], np.round(initial_guesses[1:],2))) # Initial guess for Kappa and Density
        x0 = get_initial_guesses_near_base(idx, log_posterior, prior, np.array(initial_guesses_all), n_chains=MCMC_SETTINGS['chains'])

        # setup parameter transformation:
        transform = pints.LogTransformation(n_parameters=6)

        # setup MCMC controller:
        mcmc = pints.MCMCController(log_posterior, MCMC_SETTINGS['chains'], x0, method=pints.DreamMCMC, transformation=transform)
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