
# config.py

# This file sets up data for a given CCN obs window (idx), 
# and sets the general MCMC settings.

import pandas as pd
import numpy as np
import os
from inv_ccn_utils import make_EXTRA
from pints.io import save_samples, load_samples
import glob

input_dir = '/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/input_data/' #input data
output_dir = '/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/chains/' #print chains here

base_fname = '40k_m2_logparams'  # Base filename for saving MCMC results

restart_dir = 'm2_40k_logparams' #folder with existing chains to restart from

MCMC_SETTINGS = {
'max_iterations': 40000,  # Maximum number of MCMC iterations
'burn_in': 15000,     # Number of initial phase iterations
'chains': 5,         # Number of MCMC chains
'restart': False,  # Whether to restart from existing chains
}


def load_data(idx):
    '''
    Create data structures for the MCMC run, given window index.

    Data files::
        - 'CCN': CCN observations (number concentration at 5 super saturation ratios)
        - 'M_org1_initialguess': Initial guesses fraction of organics in Aitken mode.
        - 'bimodal_params_windows_iqr': Initial guesses and ranges for bimodal parameters  (fitted to aerosol obs within each window)
        - 'mass': total aerosol mass calculated from median of the fitted bimodal parameters
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
    M_org1_initial = pd.read_csv(os.path.join(input_dir, 'M_org1_initialguess.csv'), skiprows=lambda x: x != 0 and x != idx+1, nrows=1)
    bimodal_params = pd.read_csv(os.path.join(input_dir, 'bimodal_params_windows_iqr.csv'), skiprows=lambda x: x != 0 and x != idx+1, nrows=1)
    mass = pd.read_csv(os.path.join(input_dir, 'total_mass_median_NSDparams.csv'), skiprows=lambda x: x != 0 and x != idx+1, nrows=1)
    mass_range = pd.read_csv(os.path.join(input_dir, 'mass_highres_range.csv'), skiprows=lambda x: x != 0 and x != idx+1, nrows=1)
    
    initial_guesses = [M_org1_initial['M_org1'].values[0], # fraction of organics in Aitken mode
        bimodal_params['mode1_d'].values[0], # mode1_d median
        bimodal_params['NSD1_sum'].values[0], # NSD1_sum median
        bimodal_params['mode2_d'].values[0], # mode2_d median
        bimodal_params['NSD2_sum'].values[0],  # NSD2_sum median
    ]
    
    model_inputs = [
        bimodal_params['mode1_sigma'].values[0], # mode1_sigma
        bimodal_params['mode2_sigma'].values[0], # mode2_sigma
        mass['total_mass'].values[0], # total aerosol mass (calculated with median of fitted bimodal parameters)
        mass_range['min mass (ug/m3)'].values[0], # min total aerosol mass (calculated with fitted bimodal parameters)
        mass_range['max mass (ug/m3)'].values[0], # max total aerosol mass (calculated with fitted bimodal parameters)
        mass['aitken_mass'].values[0], # aitken mass median (calculated with median of fitted bimodal parameters)
        mass['accum_mass'].values[0], # accumulation mass median (calculated with median of fitted bimodal parameters)
    ]

    prior_params = {'medians': [bimodal_params['mode1_d'].values[0], bimodal_params['NSD1_sum'].values[0], bimodal_params['mode2_d'].values[0], bimodal_params['NSD2_sum'].values[0]],
        'mad': [bimodal_params['mode1_d_mad'].values[0], bimodal_params['NSD1_sum_mad'].values[0], bimodal_params['mode2_d_mad'].values[0], bimodal_params['NSD2_sum_mad'].values[0]],
        'min': [bimodal_params['mode1_d_min'].values[0], bimodal_params['NSD1_sum_min'].values[0], bimodal_params['mode2_d_min'].values[0], bimodal_params['NSD2_sum_min'].values[0]],
        'max': [bimodal_params['mode1_d_max'].values, bimodal_params['NSD1_sum_max'].values, bimodal_params['mode2_d_max'].values, bimodal_params['NSD2_sum_max'].values],
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


def get_initial_samples(idx, posterior, base_values, num_samples, perturbation=0.2):
    base_values = np.asarray(base_values)
  
    samples = []
    attempts = 0
    max_attempts = 1000

    # Try base values first
    if np.isfinite(posterior(base_values)):
        samples.append(base_values)
    else:
        print(f"Base values for window {idx} yielded invalid posterior.")

    # Perturb base values to generate additional starting points
    while len(samples) < num_samples and attempts < max_attempts:
        # Independent perturbation per parameter
        perturb_factors = np.random.uniform(1 - perturbation, 1 + perturbation, size=base_values.shape)
        perturbed = base_values * perturb_factors

        if np.isfinite(posterior(perturbed)):
            samples.append(perturbed)
        attempts += 1

    if len(samples) < num_samples and attempts == max_attempts:
        raise RuntimeError(f"Could not generate enough valid initial guesses for window {idx} after {attempts} attempts.")

    return samples

import numpy as np

def get_initial_guesses_near_base(idx, posterior, prior, base_values, n_chains=MCMC_SETTINGS['chains'], 
                                  perturbation=0.2, max_attempts=1000):
    """
    Combo of above two functions to find initial positions for chains where base values don't automatically work.:
    1. Try base_values first.
    2. If invalid, find closest valid sample from prior.
    3. Perturb valid starting point to generate n_chains valid samples.
    """
    base_values = np.asarray(np.round(base_values,2))
    attempts = 0
    samples = []

    # Step 1: Try base values
    if np.isfinite(posterior(base_values)):
        seed = base_values
    else:
        print(f"Base values for window {idx} are invalid. Searching prior for valid alternative...")
        # Step 2: Search prior for closest valid sample
        best_sample = None
        best_distance = np.inf

        while attempts < max_attempts:
            sample = prior.sample().flatten()
            if np.isfinite(posterior(sample)):
                dist = np.linalg.norm(sample - base_values)
                if dist < best_distance:
                    best_sample = sample
                    best_distance = dist
                    if dist == 0:
                        break  # Exact match
            attempts += 1

        if best_sample is None:
            raise ValueError(f"Could not find valid sample near base values from prior for window {idx}.")
        else:
            print(f"Using closest valid sample from prior (distance={best_distance:.4f})")
            seed = best_sample

    # Step 3: Perturb seed to generate valid initial guesses
    samples.append(seed)
    attempts = 0
    while len(samples) < n_chains and attempts < max_attempts:
        perturb_factors = np.random.uniform(1 - perturbation, 1 + perturbation, size=seed.shape)
        perturbed = seed * perturb_factors
        if np.isfinite(posterior(perturbed)):
            samples.append(perturbed)
        attempts += 1

    if len(samples) < n_chains:
        raise RuntimeError(f"Could not generate {n_chains} valid initial guesses for window {idx} after {attempts} attempts.")

    return samples

def get_restart_samples(idx, n_chains):
    """
    Starting values are the final positions of existing MCMC chains for a given window index.
    """
    # Load the existing chains
    files = sorted(glob.glob(f'/proj/bolinc/users/x_maude/CCN_closure/Modal-Aerosol-Composition/chains/{restart_dir}/*_{idx}_*.csv'))

    M_org1_chains = load_samples(files[0])
    D1_chains     = load_samples(files[1])
    N1_chains     = load_samples(files[2])
    D2_chains     = load_samples(files[3])
    N2_chains     = load_samples(files[4])

    x0 = [M_org1_chains[:,-1],D1_chains[:,-1],N1_chains[:,-1],D2_chains[:,-1],N2_chains[:,-1]]
    x0_reshaped = np.stack(x0,axis=0).T
    x0_list = [row for row in x0_reshaped]

    return x0_list