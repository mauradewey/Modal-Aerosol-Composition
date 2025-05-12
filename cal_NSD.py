import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import math
import warnings
warnings.simplefilter('ignore')

import math
import numpy as np

#to construct the lognormal size distribution

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

    # Calculate dNdlogdp for all three modes and sum together
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
