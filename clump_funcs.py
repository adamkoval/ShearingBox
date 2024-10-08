# Functions pertaining to physical clump analysis and clump-scope operations
import numpy as np

import gen_funcs as gf

def calculate_density_Roche(G, Omega_K, H_g):
    """
    Calculates the Roche density. Units depend on input units. Taken from Baehr et al. 2022, Eqn. 23.
    [https://doi.org/10.3847/1538-4357/ac7228]
    In:
        > G - (flt) gravitational constant
        > Omega_K - (flt) Keplerian frequency
        > H_g - (flt) gas scale height
    """
    return np.sqrt(2 * np.pi) * 3.5 * Omega_K**2 * H_g / G