import numpy as np
import math

def execute_test_run(pars, Extra, NSD):
    """
    Executes a test run to calculate the cloud condensation nuclei (CCN) concentration
    and kappa values based on input parameters.

    Parameters:
    - pars: list or array
        Parameter values for the calculation, which include mass fractions.
    - Extra: dict
        Additional parameters such as temperature, surface tension, and particle properties.
    - NSD: numpy array
        Number size distribution for particles.

    Returns:
    - S: numpy array
        Calculated cloud condensation nuclei (CCN) concentrations.
    - k: float
        Calculated kappa value for the particles.
    - kappa_inorg: float
        Kappa value for inorganic components.
    """
    # Initialize variables to store results
    S = None
    k = None
    
    # Extract temperature and surface tension from Extra
    T = Extra['temp']        # Temperature in Kelvin
    SIGMA = Extra['sigma']   # Surface tension in N/m
    
    # Call the kappa_kohler_module to calculate CCN, kappa, and inorganic kappa
    ccn, kappa, kappa_inorg = kappa_kohler_module(Extra, NSD, pars, T, SIGMA)
    
    # Assign the results to S and k
    S = ccn
    k = kappa

    return S, k, kappa_inorg

def kappa_kohler_module(Extra, NSD, pars_temp, T, SIGMA):
    """
    Calculates cloud condensation nuclei (CCN) concentration and kappa values
    using the kappa-Köhler theory.

    Parameters:
    - Extra: dict
        Additional parameters such as densities and critical supersaturation.
    - NSD: numpy array
        Number size distribution for particles.
    - pars_temp: list or array
        Parameter values for the calculation, which include mass fractions.
    - T: float
        Temperature in Kelvin.
    - SIGMA: float
        Surface tension in N/m.

    Returns:
    - ccn: numpy array
        Calculated cloud condensation nuclei (CCN) concentrations.
    - kappa: float
        Calculated kappa value for the particles.
    - kappa_inorg: float
        Kappa value for inorganic components.
    """
    # Calculate kappa from mass concentration
    kappa = cal_kappa(Extra, pars_temp)[0]  # Total kappa value for particles
    scrit_kappa = []  # List to store critical supersaturation values for kappa

    # Extract additional parameters from Extra
    dp_dry = Extra['dp']            # Array of dry particle diameters in nm
    ss_amb = Extra['ss_amb']        # Ambient supersaturation levels in %
    d_lower_arr = Extra['d_lower_arr']  # Lower bound of diameter bins in nm
    d_upper_arr = Extra['d_upper_arr']  # Upper bound of diameter bins in nm
    wet_dia = Extra['wet_dia']      # Array of wet particle diameters in nm
    
    # Calculation of critical supersaturation from kappa-Köhler equation
    for dp in dp_dry:
        # Find index of first wet diameter greater than current dry diameter
        index = np.where(wet_dia > dp)[0][0]
        sliced_wet_dia = wet_dia[index:]  # Slice wet diameters to relevant range
        # Calculate supersaturation for each wet diameter
        SS_values = [kappa_kohler(w * 1e-9, dp * 1e-9, kappa, T, SIGMA) for w in sliced_wet_dia]
        # Store the maximum supersaturation value as the critical supersaturation
        scrit_kappa.append((max(SS_values) - 1) * 100)
    
    # Initialize CCN concentration array
    ccn = np.zeros(len(ss_amb))

    for i, ss in enumerate(ss_amb):
        try:
            act_bin = np.where(np.array(scrit_kappa) < ss)[0][0]

            ccn_x = np.sum(NSD[act_bin + 1:])  # Sum of number size distribution above activation bin
            
            # Calculate the slope for linear interpolation of activation diameter
            slope = (dp_dry[act_bin] - dp_dry[act_bin - 1]) / (scrit_kappa[act_bin] - scrit_kappa[act_bin - 1])
            dact = dp_dry[act_bin - 1] + (ss - scrit_kappa[act_bin - 1]) * slope  # Interpolated activation diameter
    
            # Calculate number of activated particles in the activation bin
            if dact > d_lower_arr[act_bin]:
                N_act_bin = NSD[act_bin] * (d_upper_arr[act_bin] - dact) / (d_upper_arr[act_bin] - d_lower_arr[act_bin])
            elif dact < d_lower_arr[act_bin]:
                multiplier = (d_upper_arr[act_bin - 1] - dact) / (d_upper_arr[act_bin - 1] - d_lower_arr[act_bin - 1])
                N_act_bin = NSD[act_bin] + NSD[act_bin - 1] * multiplier
    
            # Calculate total CCN concentration
            ccn[i] = ccn_x + N_act_bin
        except:
            ccn[i] = 0
    return ccn, kappa, cal_kappa(Extra, pars_temp)[1]

def cal_kappa(Extra, mass):
    """
    Calculate the kappa hygroscopicity parameter for the particle based on its composition.

    Parameters:
    - Extra: dict
        Additional parameters such as densities and hygroscopicity values.
    - mass: list or array
        Mass concentrations of different particle components.

    Returns:
    - kappa: list
        List containing the total kappa value for particles and the kappa for inorganics.
    """
    # Extract densities from Extra is in kg/m3
    rho_org = Extra['densities'][0]   # Density of organic material
    rho_sulp = Extra['densities'][1]   # Density of ammonium sulfate
    rho_nitr = Extra['densities'][2]  # Density of ammonium nitrate
    rho_bc = Extra['densities'][3]   # Density of black carbon
    
    # Hygroscopicity values for different components
    k_Org = Extra['kappa_org']  # Kappa for organics
    k_N = Extra['kappa_NH4NO3']             # Kappa for ammonium nitrate
    k_s = Extra['kappa_NH4SO4']  
    k_bc = Extra['eBC']  # Kappa for black carbon (assumed non-hygroscopic)

    # Calculate net hygroscopicity parameter for inorganics
    tot_vol_inorg = Extra['true_inputs'][2]/rho_sulp + Extra['true_inputs'][3]/rho_nitr
    k_inorg = k_s * (Extra['true_inputs'][2]/rho_sulp) / tot_vol_inorg + k_N * (Extra['true_inputs'][3]/rho_nitr) / tot_vol_inorg

    #calculate net inorganic density
    rho_inorg = Extra['rho_inorg']
    
    # Calculate total particle volume
    tot_vol = (
        mass[0]/rho_org +  # Volume of organic material
        mass[2]/rho_inorg + # Volume of inorganics
        mass[3]/rho_nitr + # Volume of ammonium nitrate
        mass[4]/rho_bc     # Volume of black carbon
    )

    # Calculate contributions to kappa from each component
    k1 = k_Org * (mass[0]/rho_org) / tot_vol  # Contribution from organics
    k2 = k_inorg * (mass[2]/rho_inorg) / tot_vol   # Contribution from inorganics
    k3 = k_N * (mass[3]/rho_nitr) / tot_vol   # Contribution from ammonium nitrate (is zero currently)
    k4 = k_bc * (mass[4]/rho_bc) / tot_vol

    # Total kappa value
    kappa = k1 + k2 + k3 + k4

    return [kappa, k_inorg]

def kappa_kohler(Dwet, Ddry, kappa, T, sigma):
    """
    Calculate the equilibrium supersaturation (s_eq) using the kappa-Köhler equation.

    Parameters:
    - Dwet: float
        Wet particle diameter in meters.
    - Ddry: float
        Dry particle diameter in meters.
    - kappa: float
        Hygroscopicity parameter.
    - T: float
        Temperature in Kelvin.
    - sigma: float
        Surface tension in N/m.

    Returns:
    - s_eq: float
        Equilibrium supersaturation.
    """
    Mw = 18.016 * 1e-3  # Molar mass of water in kg/mol
    R = 8.314           # Universal gas constant in J/(mol*K)
    rhow = 1000         # Density of water in kg/m^3

    # Calculate the numerator and denominator of the kappa-Köhler equation
    fact_num = Dwet**3 - Ddry**3
    fact_denum = Dwet**3 - Ddry**3 * (1 - kappa)

    # Calculate the exponential term of the kappa-Köhler equation
    exp_term = np.exp((4 * sigma * Mw) / (R * T * rhow * Dwet))

    # Calculate the equilibrium supersaturation
    s_eq = (fact_num / fact_denum) * exp_term

    return s_eq

def cal_mass(Dp, mass_frac, Extra, nsd1, nsd2):
    """
    Calculate the total mass of particles in two modes based on their size distribution,
    densities, and composition fractions.

    Parameters:
    - Dp: numpy array
        Dry particle diameters in nanometers.
    - mass_frac: list
        Mass fractions of the components in the particles: [Org, Other, NH4SO4, NH4NO3, BC].
    - Extra: dict
        Additional parameters such as densities of components.
    - nsd1: numpy array
        Number size distribution for mode 1 in cm^(-3) nm^(-1).
    - nsd2: numpy array
        Number size distribution for mode 2 in cm^(-3) nm^(-1).

    Returns:
    - mass_info: dict
        Dictionary containing the total mass and individual component masses in micrograms per cubic meter.
    """

    # Extract component densities from the 'Extra' dictionary
    rho_org = Extra['densities'][0]   # Density of organic material in kg/m^3
    rho_sulp = Extra['densities'][1]  # Density of ammonium sulfate (NH4SO4) in kg/m^3
    rho_nitr = Extra['densities'][2]  # Density of ammonium nitrate (NH4NO3) in kg/m^3
    rho_bc = Extra['densities'][3]    # Density of black carbon (BC) in kg/m^3

    # Calculate the net density of the internally mixed particles
    rho_tot = (
        mass_frac[0] * rho_org +  # Contribution from organic materials
        mass_frac[2] * rho_sulp + # Contribution from ammonium sulfate
        mass_frac[3] * rho_nitr + # Contribution from ammonium nitrate
        mass_frac[4] * rho_bc     # Contribution from black carbon
    )

    # Calculate the total mass in kg for mode 1
    mass_mode1 = rho_tot * (nsd1 * 1e6) * (4 / 3) * np.pi * (Dp * 1e-9 / 2) ** 3
    # The formula above converts number concentration to volume, and multiplies by density

    # Calculate the total mass in kg for mode 2
    mass_mode2 = rho_tot * (nsd2 * 1e6) * (4 / 3) * np.pi * (Dp * 1e-9 / 2) ** 3
    # Similar conversion as mode 1, but using mode 2 number size distribution

    # Calculate the total mass for both modes
    total_mass = sum(mass_mode2) + sum(mass_mode1)

    # Convert total and component masses to micrograms per cubic meter
    mass_info = {
        'total_mass': total_mass * 1e9,
        'tot_mass_mode1': sum(mass_mode1) * 1e9,
        'tot_mass_mode2': sum(mass_mode2) * 1e9,
        'M_org1': mass_frac[0] * sum(mass_mode1) * 1e9,
        'M_AS1': mass_frac[2] * sum(mass_mode1) * 1e9,
        'M_AN1': mass_frac[3] * sum(mass_mode1) * 1e9,
        'M_BC1': mass_frac[4] * sum(mass_mode1) * 1e9,
        'M_org2': mass_frac[0] * sum(mass_mode2) * 1e9,
        'M_AS2': mass_frac[2] * sum(mass_mode2) * 1e9,
        'M_AN2': mass_frac[3] * sum(mass_mode2) * 1e9,
        'M_BC2': mass_frac[4] * sum(mass_mode2) * 1e9,
        'tot_org': mass_frac[0] * total_mass * 1e9,
        'tot_AS': mass_frac[2] * total_mass * 1e9,
        'tot_AN': mass_frac[3] * total_mass * 1e9,
        'tot_BC': mass_frac[4] * total_mass * 1e9
    }

    return mass_info


def nsd_lognormal(dp, conc, gsd, bins):
    multiplier = conc / (math.sqrt(2 * math.pi) * np.log10(gsd))
    num = (np.log10(bins) - np.log10(dp))**2
    denum = 2 * (np.log10(gsd))**2
    return multiplier * np.exp(-(num / denum))


def size_distribution(modes, Dp):
    "Input all diameters in nanometer"
    # Create an array of zeros for the initial mode_sum
    mode_sum = np.zeros(np.shape(Dp))

    # Extract variables for each mode
    diam = modes[:, 2]
    conc = modes[:, 0]
    gsd = modes[:, 1]

    # Calculate dNdlogdp for both modes and sum together
    for j in range(0, len(diam)):
        mode = nsd_lognormal(diam[j] * 1e-9, conc[j], gsd[j], Dp * 1e-9)
        mode_sum += mode
    #Now, mode_sum contains the sum of the distributions for the two modes

    # calculate the absolute number concentration
    Dpnm = Dp   #from ra
    dlogdp_mvec = np.diff(np.log10(Dp))
    dlogdp = np.mean(dlogdp_mvec)

    mode_sum_abs = mode_sum*dlogdp
    return mode_sum, mode_sum_abs


# MAKE "EXTRA" parameter dictionary, which contains additional parameters for the calculations
# inputs are dry particle diameters
def make_EXTRA(dp_dry):

    sigma_ls = [72.8, 30, 35, 40, 45, 50, 60, 70] #in mN/m, surface tension

    # ------ create "Extra" dictionary to store additional parameters:
    Extra = dict()

    #'densities sequence: Organics, Ammonium Sulphate, Ammonium Nitrate, Black Carbon '
    Extra['densities'] = [1.50*1e3, 1.77*1e3, 1.71*1e3, 1.77*1e3]# %cross et al. 2015, %Park et al.,2004

    Extra['kappa_org'] = 0.12  # Kappa value (hygroscopicity parameter) for organic compounds
    Extra['kappa_NH4SO4'] = 0.61  # Kappa value (hygroscopicity parameter) for ammonium sulfate (NH4)2SO4
    Extra['kappa_NH4NO3'] = 0.67  # Kappa value (hygroscopicity parameter) for ammonium nitrate NH4NO3
    Extra['eBC'] = 0  # Kappa value (hygroscopicity parameter) for black carbon

    Extra['sigma'] = sigma_ls[0] * 1e-3  # Surface tension value in N/m, converted from the given sigma list in milli-Newtons/meter
    Extra['ss_amb'] = [0.1, 0.2, 0.3, 0.5, 1.0]  # Ambient supersaturation values as in the CCN counter (in percentage), CCN calculations
    Extra['wet_dia'] = np.logspace(0, 4.35, 200)  # Wet diameter of particles in nanometers, logarithmically spaced between 1 nm and 10^4.35 nm over 200 points
    Extra['temp'] = 298.48  # Median temperature in Kelvin, the hut median temperature over the entire timeseries (approximately 25.33°C)

    # Get dry particle diameters: 
    Extra['dp'] = dp_dry

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

    return Extra

