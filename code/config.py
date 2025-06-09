
# config.py

# This file sets up data for a given CCN obs window (idx), 
# and sets the general MCMC settings.

import pandas as pd
import numpy as np
import os
from inv_ccn_utils import make_EXTRA
from pints.io import save_samples

input_dir = '/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/input_data/'
output_dir = '/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/chains/'

base_fname = '30k_m2_logparams'  # Base filename for saving MCMC results

MCMC_SETTINGS = {
'max_iterations': 30000,  # Number of MCMC iterations
'burn_in': 5000,     # Number of burn-in iterations
'chains': 5,         # Number of MCMC chains
}


def load_data(idx):
    '''
    Create data structures for the MCMC run, given window index.

    Data files::
        - 'ccn_obs': CCN observations (number concentration at 5 super saturation ratios)
        - 'M_org1_initial': Initial guesses fraction of organics in Aitken mode.
        - 'bimodal_params': Initial guesses and ranges for bimodal parameters  (fitted to aerosol obs within each window)
        - 'median_mass': total aerosol mass calculated from median of the fitted bimodal parameters
        - 'mass_range': range of total aerosol mass calculated from the fitted bimodal parameters
        - 'dp_dry': dry diameters of the particles
        - 'comp_obs': observed total mass and ACSM mass fractions

    Returns:
        -model_inputs: [mode1 GSD, mode2 GSD, aerosol mass median, min, max, aitken mass median, accumulation mass median] (GSD are not optmized, aserosol mass is for mass conservation)
        -initial_guesses: [M_org1, mode1_d, mode1_N, mode2_d, mode2_N] (optmization parameters: mass fraction of organics in Aitken mode, median diameter and number concentration of the two modes)
        -prior_params: [median, MAD, min, max] for the bimodal parameters (used to set up the priors)
        -response: [CCN obs] (number concentration at 5 super saturation ratios)
    '''

    # load data:
    response = pd.read_csv(os.path.join(input_dir, 'CCN.csv'), header=None, skiprows=idx+1, nrows=1).drop(columns=0).values[0]
    M_org1_initial = pd.read_csv(os.path.join(input_dir, 'M_org1_initialguess.csv'),header=None, skiprows=idx+1, nrows=1).squeeze()
    bimodal_params = pd.read_csv(os.path.join(input_dir, 'bimodal_params_windows.csv'),header=None, skiprows=idx+1, nrows=1).drop(columns=0).values[0]
    mass = pd.read_csv(os.path.join(input_dir, 'total_mass_median_NSDparams.csv'), header=None, skiprows=idx+1, nrows=1).drop(columns=0).squeeze()
    mass_range = pd.read_csv(os.path.join(input_dir, 'mass_highres_range.csv'), header=None, skiprows=10+1, nrows=1).drop(columns=2).values[0]
    
    initial_guesses = [M_org1_initial, # fraction of organics in Aitken mode
        bimodal_params[1], # mode1_d median
        bimodal_params[7], # NSD1_sum median
        bimodal_params[4], # mode2_d median
        bimodal_params[8],  # NSD2_sum median
    ] 
    
    model_inputs = [
        bimodal_params[2], # mode1_sigma
        bimodal_params[5], # mode2_sigma
        mass[1], # total aerosol mass (calculated with median of fitted bimodal parameters)
        mass_range[0], # min total aerosol mass (calculated with fitted bimodal parameters)
        mass_range[1], # max total aerosol mass (calculated with fitted bimodal parameters)
        mass[2], # aitken mass median (calculated with median of fitted bimodal parameters)
        mass[3], # accumulation mass median (calculated with median of fitted bimodal parameters)
    ]

    prior_params = {'medians': [bimodal_params[1], bimodal_params[7], bimodal_params[4], bimodal_params[8]],
                    'mad': [bimodal_params[11], bimodal_params[17], bimodal_params[14], bimodal_params[20]],
                    'min': [bimodal_params[10], bimodal_params[16], bimodal_params[13], bimodal_params[19]],
                    'max': [bimodal_params[9], bimodal_params[15], bimodal_params[12], bimodal_params[18]],
    }

    return model_inputs, initial_guesses, prior_params, response


def get_Extra(idx):
    '''
    Get the Extra dictionary for the MCMC run.

    Returns:
        Extra dictionary containing the dry diameters of the particles and the densities of the components.
    '''

    # load data for the i-th window:
    dp_dry = np.loadtxt(os.path.join(input_dir, 'Dp.txt'))
    comp_obs = pd.read_csv(os.path.join(input_dir, 'comp.csv'), header=None, skiprows=idx+1, nrows=1).drop(columns=0).values[0].tolist()

    Extra = make_EXTRA(dp_dry)

    mass_frac = [
        comp_obs[0], # Org
        comp_obs[4], # total_mass
        comp_obs[1], # NH4SO4
        comp_obs[2], # NH4NO3
        comp_obs[3], # eBC880
        ]

    # mass vectors:
    mass_vec_NH4SO4 = comp_obs[1] * comp_obs[4]
    mass_vec_NH4NO3 = comp_obs[2] * comp_obs[4]

    # mass fractions
    mass_frac_vec_NH4SO4 = mass_vec_NH4SO4 / (mass_vec_NH4SO4 + mass_vec_NH4NO3)
    mass_frac_vec_NH4NO3 = mass_vec_NH4NO3 / (mass_vec_NH4SO4 + mass_vec_NH4NO3)

    # densities:
    rho_sulp = Extra['densities'][1]   # in kg/m^3
    rho_nitr = Extra['densities'][2]

    # get inorganic density (we include both NH4SO4 and NH4NO3)
    rho_inorg = (mass_frac_vec_NH4SO4 * rho_sulp) + (mass_frac_vec_NH4NO3 * rho_nitr)

    # add to Extra the variables that are calculated each time step:
    Extra['true_inputs'] = mass_frac
    Extra['rho_inorg'] = rho_inorg

    return Extra


def save_chain_results(samples, nchains, idx):
    """
    Save the MCMC chain results to a CSV file.

    Args:
        samples: MCMC samples.
        idx: Index of the window.
    """
    # Save the MCMC samples to a CSV file
    filename = f'mcmc_{base_fname}_{nchains}chains_{idx}.csv'
    save_samples(os.path.join(output_dir, filename), samples[:,:,0], samples[:,:,1], samples[:,:,2], samples[:,:,3], samples[:,:,4])
    print(f"Saved MCMC {idx} samples to {filename}")


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


def get_initial_samples(idx, posterior, base_values, num_samples, perturbation=0.1):
    base_values = np.asarray(base_values)
  
    samples = []
    attempts = 0
    max_attempts = 500

    while len(samples) < num_samples and attempts < max_attempts * num_samples:
        factor = np.random.uniform(1-perturbation, 1+perturbation)
        test = base_values * factor
        if np.isfinite(posterior(test)):
            samples.append(test)
        attempts += 1

    if len(samples) < num_samples:
        raise RuntimeError(f"Could not generate {num_samples} valid initial guesses for window {idx}.")

    return samples
