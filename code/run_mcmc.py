
# run_mcmc.py

# This file runs the DREAM Monte Carlo Markov Chain (MCMC) simulation for CCN closure.
# Here we choose the specific model, likelihood, and prior classes.
# Make sure that MCMC settings are set in the config.py file.

from models import CCNmodel_m2
from likelihoods import KnownSigmaGaussianLogLikelihood
from priors import joint_CauchyPrior
from config import get_Extra, load_data, get_restart_samples, get_initial_guesses_near_base, save_chain_results
import pdb

def run_mcmc_for_CCNwindow(idx):

    base_fname = '40k_m2_logparams'  # Base filename for saving MCMC results

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

        # setup model:
        m = CCNmodel_m2(Extra, model_data)

        # setup priors:
        prior = joint_CauchyPrior(prior_params, initial_guesses[0])  # M_org1_initial_guess is the first element of initial_guesses

        # setup posterior:
        log_posterior = pints.LogPosterior(
            KnownSigmaGaussianLogLikelihood(m, response),
            prior 
        )

        # get starting parameter values:
        if MCMC_SETTINGS['restart']:
            print(f"Restarting MCMC for CCN window {idx} from existing chains...")
            x0 = get_restart_samples(idx, MCMC_SETTINGS['chains'])
            
        if MCMC_SETTINGS['restart'] == False:
            print(f"Running MCMC for CCN window {idx}...")
            x0 = get_initial_guesses_near_base(idx, log_posterior, prior, np.array(initial_guesses), n_chains=MCMC_SETTINGS['chains'])
 

        # setup parameter transformation:
        transform = pints.LogTransformation(n_parameters=5)

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


