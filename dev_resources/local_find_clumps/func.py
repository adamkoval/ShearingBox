import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit

m_part = 1.44e-5 # particle mass [code units]

def get_com_coords(fin_ranked):
    """
    Reads in ranked file and returns coordinates of centre-of-mass of clumps according to ranked
    In:
        > fin_ranked - (str) path to ranked file
    Out:
        > com_coords - (arr) xyz coordinates of centre-of-mass of clump (currently just for one COM)
    """
    ranked = pd.read_csv(fin_ranked, sep='\t', skiprows=1, names=["idx", "radius", "x", "y", "z"])
    com = ranked.iloc[0]
    com_coords = com[['x', 'y', 'z']].values
    return ranked, com_coords


def read_psliceout(fin_raw):
    """
    Reads in PENCIL slice data file and returns xyz coordinates of all particles in slice
    In:
        > fin_raw - (str) path to psliceout file
    Out:
        > raw_coords - (arr) xyz coordinates of all particles in slice
    """
    raw = pd.read_csv(fin_raw, sep='\s+', index_col=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'size'], header=None)
    raw_coords = raw[['x', 'y', 'z']].values
    raw_vels = raw[['vx', 'vy', 'vz']].values
    raw_sizes = raw[['size']].values
    raw_idx = raw.index.values
    return raw_coords, raw_vels, raw_sizes, raw_idx


@jit(nopython=True)
def compute_com_densities(com_coords, raw_coords, clump_rad, dr):
    """
    Computes the radial distances of coords from com and returns an ordered array
    In:
        > com_coords - (arr) xyz coordinates of centre-of-mass of clump (currently just for one COM)
        > raw_coords - (arr) xyz coordinates of all particles in slice
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
        numpart[i] = np.sum(distances < clump_rad[i])
        shell_volume = (4.0 / 3.0) * np.pi * (clump_rad[i]**3 - (clump_rad[i] - dr)**3)
        if i == 0:
            clump_dens[i] = numpart[i] * m_part / shell_volume
        else:
            clump_dens[i] = (numpart[i] - numpart[i-1]) * m_part / shell_volume
    return clump_dens, numpart, distances


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


def sort_distances(dr, n_shells, com_coords, raw_coords, raw_idx, var_shell=True):
    """
    Obtains a number of clump properties
    In:
        > dr - (float) shell width (or zeroth radius for variable shells)
        > n_shells - (int) number of shells
        > com_coords - (arr) xyz coordinates of centre-of-mass of clump (currently just for one COM)
        > raw_coords - (arr) xyz coordinates of all particles in slice
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


class analysis_plot:
    """
    Class containing analysis plots
    """
    def __init__(self, clump_rad, clump_dens, numpart, sorted_distances):
        """
        Initialise class
        In:
            > clump_rad - (NParr) list of shell radii
            > clump_dens - (NParr) densities per shell
            > numpart - (NParr) number of particles per shell
            > sorted_distances - (NParr) sorted distances between particles from smallest to largest
        """
        # Definitions
        self.clump_rad = clump_rad
        self.clump_dens = clump_dens
        self.numpart = numpart
        self.sorted_distances = sorted_distances

        # Masking
        self.xlim = 2e-4
        mask1 = (self.clump_rad < self.xlim)
        self.masked_clump_rad = self.clump_rad[mask1]
        self.masked_numpart = self.numpart[mask1]
        self.lim_numpart = np.max(self.masked_numpart)

    def create(self):
        """
        Creates the figure
        """
        self.fig, self.ax = plt.subplots(1, 3, figsize=(10,5))
        self.add_dens_rad_plot()
        self.add_num_rad_plot()
        self.add_mean_dist_num_plot()
        plt.tight_layout()
        plt.show()
    
    def add_dens_rad_plot(self):
        """
        Adds density(rad) plot
        """
        self.ax[0].plot(self.clump_rad, self.clump_dens)
        self.ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        self.ax[0].set_xlabel("Radius [c.u.]")
        self.ax[0].set_ylabel("Density [c.u.]")
        self.ax[0].set_xlim(0, self.xlim)

    def add_num_rad_plot(self):
        """
        Adds partnum(rad) plot
        """
        self.ax[1].plot(self.masked_clump_rad, self.masked_numpart)
        self.ax[1].set_xlabel("Radius [c.u.]")
        self.ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        self.ax[1].set_ylabel("Num part")
        self.ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        self.ax[1].axhline(self.lim_numpart, c='r', label=f"npart = {self.lim_numpart}")
        # self.ax[1].text(f"{lim_numpart}", 0.0, lim_numpart)
        self.ax[1].set_xlim(0, self.xlim)
        self.ax[1].legend()

    def add_mean_dist_num_plot(self):
        """
        Adds mean_dist(numpart) plot
        """
        masked_mean_dists = self.mean_dists(self.sorted_distances[:int(self.lim_numpart)])
        masked_numparts = range(len(self.sorted_distances[:int(self.lim_numpart)]))
        self.ax[2].plot(masked_mean_dists, masked_numparts)
        self.ax[2].set_xlabel("Mean distance (c.u.)")
        self.ax[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        self.ax[2].set_ylabel("Num part")
        self.ax[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    @staticmethod
    @jit(nopython=True)
    def mean_dists(sorted_distances):
        """
        Finds mean distances between particles at each radius
        In:
            > sorted_distances - (NParr) sorted distances between particles from smallest to largest
        Out:
            > means - (NParr) mean distances between particles as a function of the number of particles
        """
        cumulative_sums = np.cumsum(sorted_distances)
        means = cumulative_sums / np.arange(1, len(sorted_distances) + 1)
        return means


# def R_H(M_c, rho_m0):
#     return (M_c / (24 * rho_m0))**(1/3)

# def find_R_Hs(numpart, rho_m0):
#     return [R_H(part*m_part, rho_m0) for part in numpart]

def R_H(M_c, Omega, G):
    return (M_c / (3 * Omega**2) * G)**(1/3)

def find_R_Hs(numpart, Omega, G):
    return [R_H(part*m_part, Omega, G) for part in numpart]

def Toomre_Q(c_s, kappa, G, Sigma):
    return (c_s * kappa) / (np.pi * G * Sigma)