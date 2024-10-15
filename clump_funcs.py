# Functions pertaining to physical clump analysis and clump-scope operations
import numpy as np
import pandas as pd

from numba import jit

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


def R_H(M_c, Omega, G):
    """
    Calculates the Hill radius. Units depend on input units. Taken from Baehr et al. 2022, Eqn. 24.
    [https://doi.org/10.3847/1538-4357/ac7228]
    In:
        > M_c - (flt) clump mass
        > Omega - (flt) Keplerian frequency
        > G - (flt) gravitational constant
    """
    return (M_c / (3 * Omega**2) * G)**(1/3)


def Toomre_Q(c_s, kappa, G, Sigma):
    """
    Calculates the Toomre Q parameter. Units depend on input units.
    In:
        > c_s - (flt) sound speed
        > kappa - (flt) epicyclic frequency
        > G - (flt) gravitational constant
        > Sigma - (flt) surface density
    """
    return (c_s * kappa) / (np.pi * G * Sigma)


def sort_distances(dr, n_shells, com_coords, raw_coords, raw_idx, var_shell=True):
    """
    Obtains a number of clump properties
    In:
        > dr - (float) shell width (or zeroth radius for variable shells)
        > n_shells - (int) number of shells
        > com_coords - (NParr) xyz coordinates of centre-of-mass of clump (currently just for one COM)
        > raw_coords - (NParr) xyz coordinates of all particles in slice
        > raw_idx - (NParr) indices of all particles in slice
        > var_shell - (BOOL) turn on/off variable shell width
    Out:
        > clump_rad - (NParr) list of shell radii
        > clump_dens - (NParr) densities per shell
        > numpart - (NParr) number of particles per shell
        > distances - (NParr) distances of all particles from COM
        > sorted_distances - (NParr) sorted distances between particles from smallest to largest
    """
    if not var_shell:
        clump_rad = np.arange(dr, dr * n_shells, dr)
    else:
        clump_rad = np.array(var_r(n_shells, dr))
    clump_dens, numpart, distances = compute_com_densities(com_coords, raw_coords, clump_rad, dr)
    dist_df = pd.DataFrame(distances, index=raw_idx)
    sorted_distances = np.array(dist_df.sort_values(0))
    return clump_rad, clump_dens, numpart, distances, sorted_distances


@jit(nopython=True)
def var_r(n_shells, R0):
    """
    Function to define variable-width shells to keep constant volume in each shell
    In:
        > n_shells - (int) number of shells
        > R0 - (float) initial radius
    Out:
        > cump_rad - (arr) list of shell radii
    """
    cump_rad = [0] # initialise with zero as "1st" radius, i.e., first inner radius
    Rprev = R0
    cump_rad.append(Rprev)
    for n in range(n_shells):
        t = (Rprev**3 + R0**3)**(1/3) - Rprev
        Rprev = Rprev + t
        cump_rad.append(Rprev)
    return cump_rad


@jit(nopython=True)
def compute_com_densities(com_coords, raw_coords, clump_rad, dr):
    """
    Computes the radial distances of coords from com and returns an ordered array
    In:
        > com_coords - (NParr) xyz coordinates of centre-of-mass of clump (currently just for one COM)
        > raw_coords - (NParr) xyz coordinates of all particles in slice
        > clump_rad - (arr) radii delimiting spherical shells for integration
        > dr - (float) shell width (or zeroth radius for variable shells)
    Out:
        > clump_dens - (NParr) densities per shell
        > numpart - (NParr) number of particles per shell
        > distances - (NParr) distances of all particles from COM
    """
    # Initialise values, shells and part & dens per shell
    numpart = np.zeros(len(clump_rad))
    clump_dens = np.zeros(len(clump_rad))

    # Compute distances
    distances = np.sqrt((raw_coords[:, 0] - com_coords[0])**2 + (raw_coords[:, 1] - com_coords[1])**2 + (raw_coords[:, 2] - com_coords[2])**2)

    # Compute particle counts and densities per radial bin
    for i in range(len(clump_rad)):
        numpart[i] = int(np.sum(distances < clump_rad[i]))
        shell_volume = (4.0 / 3.0) * np.pi * (clump_rad[i]**3 - (clump_rad[i] - dr)**3)
        if i == 0:
            clump_dens[i] = numpart[i] * gf.m_swarm / shell_volume
        else:
            clump_dens[i] = (numpart[i] - numpart[i-1]) * gf.m_swarm / shell_volume
    return clump_dens, numpart, distances