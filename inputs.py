sigma_ls = [72.8, 30, 35, 40, 45, 50, 60, 70] #in mN/m, surface tension

'densities sequence: Organics, Ammonium Sulphate, Ammonium Nitrate, Black Carbon '
Extra['densities'] = [1.50*1e3, 1.77*1e3, 1.71*1e3, 1.77*1e3]# %cross et al. 2015, %Park et al.,2004

rho_org = Extra['densities'][0]
rho_sulp = Extra['densities'][1]   # in kg/m^3
rho_nitr = Extra['densities'][2]
rho_bc = Extra['densities'][3]
                    

Extra['kappa_org'] = 0.1  # Kappa value (hygroscopicity parameter) for organic compounds
Extra['kappa_NH4SO4'] = 0.61  # Kappa value (hygroscopicity parameter) for ammonium sulfate (NH4)2SO4
Extra['kappa_NH4NO3'] = 0.67  # Kappa value (hygroscopicity parameter) for ammonium nitrate NH4NO3
Extra['eBC'] = 0  # Kappa value (hygroscopicity parameter) for black carbon

Extra['sigma'] = sigma_ls[0] * 1e-3  # Surface tension value in N/m, converted from the given sigma list in milli-Newtons/meter
Extra['ss_amb'] = [0.1, 0.2, 0.3, 0.5, 1.0]  # Ambient supersaturation values as in the CCN counter (in percentage), CCN calculations
Extra['wet_dia'] = np.logspace(0, 4.35, 200)  # Wet diameter of particles in nanometers, logarithmically spaced between 1 nm and 10^4.35 nm over 200 points
Extra['temp'] = 298.48  # Median temperature in Kelvin, the hut median temperature over the entire timeseries (approximately 25.33°C)

# Calculate the initial upper and lower boundaries for the first dry particle diameter (dp_dry)
d_upper0 = dp_dry[0] + (dp_dry[1] - dp_dry[0]) / 2  # Upper boundary for the first bin
d_lower0 = dp_dry[0] - (dp_dry[1] - dp_dry[0]) / 2  # Lower boundary for the first bin

# Initialize arrays to store the lower and upper boundaries of all dry particle diameters
# Using np.full_like to create arrays of the same shape as dp_dry, filled with the initial lower/upper values
d_lower_arr_vec = np.full_like(dp_dry, d_lower0)  # Array for lower boundaries, starting with d_lower0
d_upper_arr_vec = np.full_like(dp_dry, d_upper0)  # Array for upper boundaries, starting with d_upper0

# Populate the arrays with the upper and lower diameter boundaries for each particle size
for i in range(1, len(dp_dry)):
    d_lower_arr_vec[i] = d_upper_arr_vec[i - 1]  # Lower boundary for the current bin equals the upper boundary of the previous bin
    d_upper_arr_vec[i] = dp_dry[i] + dp_dry[i] - d_lower_arr_vec[i]  # Upper boundary calculated based on particle diameter and lower boundary

# Store the lower and upper boundary arrays in the 'Extra' dictionary for later use
Extra['d_lower_arr'] = d_lower_arr_vec  # Lower boundary array
Extra['d_upper_arr'] = d_upper_arr_vec  # Upper boundary array

# Load observed data from CSV files
# nsd_abs1 and nsd_abs2: Number Size Distribution (NSD) data for mode 1 and mode 2
# comp_obs: Composition observation data
# ccn_obs: Cloud Condensation Nuclei (CCN) observation data
nsd_abs1 = pd.read_csv(os.path.join(obs_dir, 'NSD_mode1.csv'))  # Load NSD data for mode 1
nsd_abs2 = pd.read_csv(os.path.join(obs_dir, 'NSD_mode2.csv'))  # Load NSD data for mode 2
comp_obs = pd.read_csv(os.path.join(obs_dir, 'comp.csv'))  # Load observed composition data, mass fractions/concentrations
ccn_obs = pd.read_csv(os.path.join(obs_dir, 'CCN.csv'))  # Load observed CCN data

