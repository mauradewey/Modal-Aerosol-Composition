
# priors.py

# This file contains the prior distributions for the CCN closure model.
# It sets up the joint prior, using the same base across all bimodal parameters.

# At the moment, the base options are:
# - GaussianLogPrior
# - CauchyLogPrior
# - UniformLogPrior
# - a combination of the above with a uniform prior for the observational noise (sigma)
# - in all cases, the M_org1 parameter has a half-Cauchy prior.

import pints
import numpy as np


# A uniform prior for M_org1
M_org1_uniform = pints.UniformLogPrior(0, 10.0)  # Uniform prior for M_org1

# priors for bimodal parameters (D1, N1, D2, N2):
def joint_GaussianPrior(prior_params):
    """
    Args:
        prior_params (dict): Dictionary containing the median NSD parameter values, so the priors are centered
        on the initial guesses. Std is set to half the initial guess.
    """
    # unpack:
    medians = prior_params['medians']
    # Create a Gaussian log prior for each parameter truncated to be positive:
    priors = [pints.TruncatedGaussianLogPrior(medians[i], medians[i]*0.5, 0.0, np.inf) for i in range(len(prior_params))]
    
    # Combine into a single log-prior
    return pints.ComposedLogPrior(M_org1_prior, *priors)


def joint_CauchyPrior(prior_params, Morg1_initial_guess):
    """
    Args:
        prior_params (dict): Dictionary containing the median and MAD of the NSD parameter values, so the priors are centered
        on the initial guesses. Scale is set to the minimum of 1 or the median absolute deviation.
    """
    # unpack:
    medians = np.round(prior_params['medians'],4)
    mad = np.round(prior_params['mad'],4)

    # Create a Cauchy log prior for each parameter which is truncated to be positive:
    M_org1_prior = pints.HalfCauchyLogPrior(np.round(Morg1_initial_guess,4), 0.5) 
    priors = [pints.HalfCauchyLogPrior(medians[i], min(1, mad[i])) for i in range(len(medians))]
    
    # Combine into a single log-prior
    return pints.ComposedLogPrior(M_org1_prior, *priors)


def joint_AllUniformPrior(prior_params):
    """
    Args:
        prior_params (dict): Dictionary containing the min and max of the NSD parameter values, so the priors are centered
        on the initial guesses. Scale is set to the median absolute deviation.
    """
    # unpack:
    min_vals = prior_params['min']
    max_vals = prior_params['max']

    # Create a Uniform log prior for each parameter
    priors = [pints.UniformLogPrior(max(min_vals[i]-10,0), max_vals[i]+10) for i in range(len(min_vals))]
    
    # Combine into a single log-prior
    return pints.ComposedLogPrior(M_org1_uniform, *priors)


def joint_SigmaPrior(param_prior, response):
    """
    Add a uniform log-prior for the observational noise (sigma) to the joint prior.

    Args:
        param_prior: Prior for the bimodal parameters (M_org1, D1, N1, D2, N2).
        response: Observed data for the model.
    """
    # Uniform log-prior for observational noise (between 0 and 0.1*observed data)
    low = 0.0 # Lower bound for sigma 
    high = np.ceil(np.array(response)*0.1) # Upper bound for sigma, 10% of observed data
    sigma_prior = [pints.UniformLogPrior(low, high[i]) for i in range(5)]  # uniform sigma prior
    
    # Combine into a single log-prior
    return pints.ComposedLogPrior(param_prior, sigma_prior)

def joint_UniformMorgCauchyPrior(prior_params):
    """
    Uniform prior for M_org1 and Cauchy priors for bimodal parameters (D1, N1, D2, N2).
    Args:
        prior_params (dict): Dictionary containing the median and MAD of the NSD parameter values, so the priors are centered
        on the initial guesses. Scale is set to the median absolute deviation.
    """
    # unpack:
    medians = np.round(prior_params['medians'],4)
    mad = np.round(prior_params['mad'],4)

    # Create a Cauchy log prior for each parameter which is truncated to be positive:
    priors = [pints.HalfCauchyLogPrior(medians[i], min(1, mad[i])) for i in range(len(medians))]
    
    # Combine into a single log-prior
    return pints.ComposedLogPrior(M_org1_uniform, *priors)

def joint_TruncatedGaussianPrior(prior_params):
    """
    Args:
        prior_params (dict): Dictionary containing the median and MAD of the NSD parameter values, so the priors are centered
        on the initial guesses. Scale is set to the median absolute deviation.
    """
    # unpack:
    medians = prior_params['medians']
    mad = prior_params['mad']
    min = prior_params['min']
    max = prior_params['max']

    # Create a truncated Gaussian log prior for each parameter which is truncated to be positive:
    priors = [pints.TruncatedGaussianLogPrior(medians[i], mad[i], max(0, min[i]), max[i]) for i in range(len(medians))]

    # Combine into a single log-prior
    return pints.ComposedLogPrior(M_org1_prior, *priors)
