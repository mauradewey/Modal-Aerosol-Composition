
# config.py

# This file sets up data for a given CCN obs window (idx), 
# and sets the general MCMC settings.

import pandas as pd
import numpy as np
from inv_ccn_utils import make_EXTRA
from pints.io import save_samples

MCMC_SETTINGS = {
    'max_iterations': 20000,  # Number of MCMC iterations
    'burn_in': 2000,     # Number of burn-in iterations
    'chains': 4,         # Number of MCMC chains
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
        -model_inputs: [mode1 GSD, mode2 GSD, aerosol mass median, min, max] (GSD are not optmized, aserosol mass is for mass conservation)
        -initial_guesses: [M_org1, mode1_d, mode1_N, mode2_d, mode2_N] (optmization parameters: mass fraction of organics in Aitken mode, median diameter and number concentration of the two modes)
        -prior_params: [median, MAD, min, max] for the bimodal parameters (used to set up the priors)
        -response: [CCN obs] (number concentration at 5 super saturation ratios)
    '''

    # load data:
    response = pd.read_csv('../input_data/CCN.csv', header=None, skiprows=idx+1, nrows=1).drop(columns=0).values[0]
    M_org1_initial = pd.read_csv('../input_data/M_org1_initialguess.csv',header=None, skiprows=idx+1, nrows=1).squeeze()
    bimodal_params = pd.read_csv('../input_data/bimodal_params_windows.csv',header=None, skiprows=idx+1, nrows=1).drop(columns=0).values[0]
    median_mass = pd.read_csv('../input_data/mass_from_median_NSDparams.csv', header=None, skiprows=idx+1, nrows=1).drop(columns=1).squeeze()
    mass_range = pd.read_csv('../input_data/mass_highres_range.csv', header=None, skiprows=10+1, nrows=1).drop(columns=2).values[0]
    
    initial_guesses = [M_org1_initial, # fraction of organics in Aitken mode
        bimodal_params[1], # mode1_d median
        bimodal_params[7], # NSD1_sum median
        bimodal_params[4], # mode2_d median
        bimodal_params[8],  # NSD2_sum median
    ] 
    
    model_inputs = [
        bimodal_params[2], # mode1_sigma
        bimodal_params[5], # mode2_sigma
        median_mass, # total aerosol mass (calculated with median of fitted bimodal parameters)
        mass_range[0], # min total aerosol mass (calculated with fitted bimodal parameters)
        mass_range[1], # max total aerosol mass (calculated with fitted bimodal parameters)
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
    dp_dry = np.loadtxt('../input_data/Dp.txt')
    comp_obs = pd.read_csv('../input_data/comp.csv', header=None, skiprows=idx+1, nrows=1).drop(columns=0).values[0].tolist()

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


def save_chain_results(samples, idx):
    """
    Save the MCMC chain results to a CSV file.

    Args:
        samples: MCMC samples.
        idx: Index of the window.
    """
    # Save the MCMC samples to a CSV file
    filename = f'../chains/chain_results_{idx}.csv'
    save_samples(filename, samples[:,:,0], samples[:,:,1], samples[:,:,2], samples[:,:,3], samples[:,:,4])
    print(f"Saved MCMC {idx} samples to {filename}")