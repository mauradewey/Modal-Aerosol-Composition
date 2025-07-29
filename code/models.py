
# models.py

# This file contains the forward model for the CCN closure calculation. 
# (maps optimization parameters -> output; M_org1 + D1 + N1 + D2 + N2 -> total CCN at 5 supersaturations)
# There are two versions, based on the mass conservation approach:
#       CCNmodel_m1 requires the total particle mass to be within 10% of the median mass calculated with the median fitted NSD parameters.
#       CCNmodel_m2 requires the total particle mass for each mode to be within 10% of the median mass for each mode calculated with the median fitted NSD parameters. 
# CCN model specific functions are imported from inv_ccn_utils.py.

import pints
import numpy as np
from inv_ccn_utils import *

class CCNmodel_m1(pints.ForwardModel):
    # Calculate CCN, given the mass fraction of organics in the aitken mode, D1, N1, D2, N2 as parameters

    def __init__(self, Extra, model_data, return_all=False):
        # Define the number of parameters and outputs:
        self._n_parameters = 5  # Number of optimization parameters
        self._n_outputs = 5 #number of outputs (CCN at 5 supersaturations)

        # Get pre-calculated/not-optimized parameters:
        self.Extra = Extra # Extra is a dictionary containing all constant parameters for the CCN closure calculation     
        self.GSD1 = model_data[0] # geometric standard deviation for mode 1
        self.GSD2 = model_data[1] # geometric standard deviation for mode 2
        self.med_mass = model_data[2] # median mass for optimization (calculated with median of fitted bimodal parameters)

        # Define the return_all flag: True, return all outputs (CCN, k_org, k_inorg, mass fractions, total masses). False, return only the total CCN (default for optimization)
        self.return_all = return_all

      
    def __call__(self, params):
        
        M_org1 = params[0]  # Mass of organics in Aitken mode (optimization parameter)
        D1 = params[1]  # Median diameter of Aitken mode (optimization parameter)
        N1 = params[2]  # Median number concentration of Aitken mode (optimization parameter)
        D2 = params[3]  # Median diameter of Accumulation mode (optimization parameter)
        N2 = params[4]  # Median number concentration of Accumulation mode (optimization parameter)


        # calculate size distributions:
        NSD1 = size_distribution(np.array([[N1, self.GSD1, D1]]), self.Extra['dp']) # Aitken mode
        NSD1_vec = NSD1[1] # Aitken mode absolute NSD
        NSD2 = size_distribution(np.array([[N2, self.GSD2, D2]]), self.Extra['dp']) # Accumulation mode
        NSD2_vec = NSD2[1] # Accumulation mode absolute NSD

        # calculate the mass of the particles in both modes:
        info_mass = cal_mass(self.Extra['dp'], self.Extra['true_inputs'], self.Extra, NSD1_vec, NSD2_vec)

        # get initial mass fractions for both modes:
        initial_M_BC1 = info_mass['M_BC1']  # Black carbon in Aitken mode
        initial_M_BC2 = info_mass['M_BC2']  # Black carbon in Accumulation mode
        tot_mass_org = info_mass['tot_org']
        tot_mass_AS = info_mass['tot_AS'] + info_mass['tot_AN']  # Total ammonium salts (sulfate + nitrate)
        initial_tot_acc_mass = info_mass['tot_mass_mode2']  # Total mass in Accumulation mode
        initial_tot_ait_mass = info_mass['tot_mass_mode1']  # Total mass in Aitken mode
 
        # Mass fractions for black carbon
        f_BC1 = initial_M_BC1 / initial_tot_ait_mass
        f_BC2 = initial_M_BC2 / initial_tot_acc_mass

        # Calculate remaining mass in each mode
        M_AS1 = initial_tot_ait_mass - (M_org1 + initial_M_BC1) # inorganics mass in mode1
        M_org2 = tot_mass_org - M_org1 #organics mass in mode2
        M_AS2 = tot_mass_AS - M_AS1 #inorganics mass in mode2
        total_ait_mass = M_org1 + M_AS1 + initial_M_BC1
        total_acc_mass = M_org2 + M_AS2 + initial_M_BC2

        # Check total mass:
        total_mass = total_ait_mass + total_acc_mass # total mass of both modes with optimized parameters

        # Ensure mass is non-negative and within tolerance
        if(
        (0.9 * self.med_mass < total_mass < 1.1 * self.med_mass) and
        M_org2 >= 0 and M_AS2 >= 0 and M_AS1 >= 0 and M_org1 >= 0
        ):
            
        # If within tolerance, continue:
            f_org1 = M_org1 / total_ait_mass
            f_AS1 = M_AS1 / total_ait_mass
            f_org2 = M_org2 / total_acc_mass
            f_AS2 = M_AS2 / total_acc_mass
   
            mass_frac_aitken = [f_org1, 1, f_AS1, 0, f_BC1] # sequence is organics, total mass fraction, inorganics, nitrate, eBC
            mass_frac_accumulation = [f_org2, 1, f_AS2, 0, f_BC2]

            # calculate CCN, k_org, k_inorg for both modes:
            ccn1, k1, k_inorg1 = execute_test_run(mass_frac_aitken, self.Extra, NSD1_vec)
            ccn2, k2, k_inorg2 = execute_test_run(mass_frac_accumulation, self.Extra, NSD2_vec)
        
            # return CCN total:
            if self.return_all:
                return ccn1, ccn2, k1, k2, k_inorg1, k_inorg2, mass_frac_aitken, mass_frac_accumulation, total_ait_mass, total_acc_mass, total_mass,info_mass, NSD1_vec, NSD2_vec
            else:
                return ccn1+ccn2
        

        # If masses not within tolerance, return None (this will be penalized in the likelihood function)
        else:
            return None
    
    def n_parameters(self):
        return self._n_parameters
    
    def n_outputs(self):
        return self._n_outputs
    


class CCNmodel_m2(pints.ForwardModel):
    # Calculate CCN, given the mass fraction of organics in the aitken mode, D1, N1, D2, N2 as parameters

    def __init__(self, Extra, model_data, return_all=False):
        # Define the number of parameters and outputs:
        self._n_parameters = 5  # Number of optimization parameters
        self._n_outputs = 5 #number of outputs (CCN at 5 supersaturations)

        # Get pre-calculated/not-optimized parameters:
        self.Extra = Extra # Extra is a dictionary containing all constant parameters for the CCN closure calculation     
        self.GSD1 = model_data[0] # geometric standard deviation for mode 1
        self.GSD2 = model_data[1] # geometric standard deviation for mode 2
        self.ait_mass = model_data[5] # aitken mass 
        self.acc_mass = model_data[6] # accumulation mass

        # Define the return_all flag: True, return all outputs (CCN, k_org, k_inorg, mass fractions, total masses). False, return only the total CCN (default for optimization)
        self.return_all = return_all

    
    def __call__(self, params):
        # unpack optimization parameters:
        M_org1 = params[0]  # Mass of organics in Aitken mode (optimization parameter)
        D1 = params[1]  # Median diameter of Aitken mode (optimization parameter)
        N1 = params[2]  # Median number concentration of Aitken mode (optimization parameter)
        D2 = params[3]  # Median diameter of Accumulation mode (optimization parameter)
        N2 = params[4]  # Median number concentration of Accumulation mode (optimization parameter)


        # calculate size distributions:
        NSD1 = size_distribution(np.array([[N1, self.GSD1, D1]]), self.Extra['dp']) # Aitken mode
        NSD1_vec = NSD1[1] # Aitken mode absolute NSD
        NSD2 = size_distribution(np.array([[N2, self.GSD2, D2]]), self.Extra['dp']) # Accumulation mode
        NSD2_vec = NSD2[1] # Accumulation mode absolute NSD

        # calculate the mass of the particles in both modes:
        info_mass = cal_mass(self.Extra['dp'], self.Extra['true_inputs'], self.Extra, NSD1_vec, NSD2_vec)

        # get initial mass fractions for both modes:
        initial_M_BC1 = info_mass['M_BC1']  # Black carbon in Aitken mode
        initial_M_BC2 = info_mass['M_BC2']  # Black carbon in Accumulation mode
        tot_mass_org = info_mass['tot_org']
        tot_mass_AS = info_mass['tot_AS'] + info_mass['tot_AN']  # Total ammonium salts (sulfate + nitrate)
        initial_tot_acc_mass = info_mass['tot_mass_mode2']  # Total mass in Accumulation mode
        initial_tot_ait_mass = info_mass['tot_mass_mode1']  # Total mass in Aitken mode
 
        # Mass fractions for black carbon
        f_BC1 = initial_M_BC1 / initial_tot_ait_mass
        f_BC2 = initial_M_BC2 / initial_tot_acc_mass

        # Calculate remaining mass in each mode
        M_AS1 = initial_tot_ait_mass - (M_org1 + initial_M_BC1) # inorganics mass in mode1
        M_org2 = tot_mass_org - M_org1 #organics mass in mode2
        M_AS2 = tot_mass_AS - M_AS1 #inorganics mass in mode2
        total_ait_mass = M_org1 + M_AS1 + initial_M_BC1
        total_acc_mass = M_org2 + M_AS2 + initial_M_BC2

        # Check total mass:
        total_mass = total_ait_mass + total_acc_mass # total mass of both modes with optimized parameters

        # Ensure mass is non-negative and within tolerance
        if(
        (0.9 * self.ait_mass < total_ait_mass < 1.1 * self.ait_mass) and
        (0.9 * self.acc_mass < total_acc_mass < 1.1 * self.acc_mass) and
        M_org2 >= 0 and M_AS2 >= 0 and M_AS1 >= 0 and M_org1 >= 0
        ):
            
        # If within tolerance, continue:
            f_org1 = M_org1 / total_ait_mass
            f_AS1 = M_AS1 / total_ait_mass
            f_org2 = M_org2 / total_acc_mass
            f_AS2 = M_AS2 / total_acc_mass
   
            mass_frac_aitken = [f_org1, 1, f_AS1, 0, f_BC1] # sequence is organics, total mass fraction, inorganics, nitrate, eBC
            mass_frac_accumulation = [f_org2, 1, f_AS2, 0, f_BC2]

            # calculate CCN, k_org, k_inorg for both modes:
            ccn1, k1, k_inorg1 = execute_test_run(mass_frac_aitken, self.Extra, NSD1_vec)
            ccn2, k2, k_inorg2 = execute_test_run(mass_frac_accumulation, self.Extra, NSD2_vec)
        
            # return CCN total:
            if self.return_all:
                return ccn1, ccn2, k1, k2, k_inorg1, k_inorg2, mass_frac_aitken, mass_frac_accumulation, total_ait_mass, total_acc_mass, total_mass,info_mass, NSD1_vec, NSD2_vec
            else:
                return ccn1+ccn2
        

        # If masses not within tolerance, return None (this will be penalized in the likelihood function)
        else:
            return None
    
    def n_parameters(self):
        return self._n_parameters
    
    def n_outputs(self):
        return self._n_outputs