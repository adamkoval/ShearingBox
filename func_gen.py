# General functions for data loading and processing
import pandas as pd
import glob
import inspect
import json
import os
import re

import numpy as np

import func_ranked as rf
import func_clump as cf
import func_plot as pf

m_swarm = 1.44e-5  # particle mass [code units]

# READ/WRITE FUNCTIONS
def read_config(path_config):
    """
    Reads in config.json file
    In:
        > path_config - (str) path to config.json file
    Out:
        > dict_config - (dct) dictionary containing config parameters
    """
    # Report function name
    print(report_function_name(), flush=True)

    # Load config
    print("\tReading config", flush=True)
    with open(path_config, "r") as f:
        dict_config = json.load(f)
    print(f"\t\tConfig file pertains to:", flush=True)
    [print(f"\t\t\t{key} = {dict_config[key]}") for key in dict_config]
    return dict_config


def read_figure_settings(fig_details_csv):
    """
    Reads in figure settings from a .csv file
    In:
        > fig_details_csv - (str) path to figure details .csv file
    Out:
        > fig_details - (df) dataframe containing figure details
    """
    details = pd.read_csv(fig_details_csv, sep=',', header=0)
    return details


def read_paths_psliceout(path_data, selected_figs):
    """
    Reads paths to psliceout files
    In:
        > path_data - (str) path to data folder
        > selected_figs - (lst) list of selected figures
    Out:
        > paths_psliceout - (dct) paths to psliceout files
    """
    paths_psliceout = {}
    for fig in selected_figs:
        paths_psliceout[fig] = glob.glob(path_data + fig + "/*.dat")
    return paths_psliceout


def load_psliceout(path_psliceout):
    """
    Reads in PENCIL slice data file and returns xyz coordinates of all particles in slice
    In:
        > path_psliceout - (str) path to psliceout file
    Out:
        > data - (dct)
            raw_coords: xyz coordinates of all particles in slice
            raw_vels: xyz velocities of all particles in slice
            raw_sizes: sizes of all particles in slice
            raw_idx: indices of all particles in slice
    """
    raw = pd.read_csv(path_psliceout, sep='\s+', index_col=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'size'], header=None).sort_index()
    raw_coords = raw[['x', 'y', 'z']].values
    raw_vels = raw[['vx', 'vy', 'vz']].values
    raw_sizes = raw['size'].values
    raw_idx = raw.index.values
    data = {'raw_coords': raw_coords, 'raw_vels': raw_vels, 'raw_sizes': raw_sizes, 'raw_idx': raw_idx}
    return data


def write_coords_com_rankedf(coords_com_rankedf, outdir):
    """
    Writes coords_com_rankedf into a file
    In:
        > coords_com_rankedf - (dct) dictionary containing COM coordinates, indices of members, and Hill radii of clumps for all ranked files
        > outdir - (str) output dir (as given in config file)
    """
    # Report function name
    print(report_function_name(), flush=True)

    # Check for/create output dir
    outdir_coords_com = outdir.strip('/') + '/' + 'com_data/'
    if not os.path.exists(outdir_coords_com):
        print(f"\tCreating directory {outdir_coords_com}", flush=True)
        os.makedirs(outdir_coords_com)

    # Enter main loop
    for _fin_ranked in coords_com_rankedf:
        # Obtain all com data from that file
        data_coms = coords_com_rankedf[_fin_ranked]['data_com']

        # Iterate through clumps
        for i_clump in range(len(data_coms['idx'])):
            fout = outdir_coords_com + _fin_ranked + "_clump_" + str(i_clump) + ".dat"
            if os.path.exists(fout):
                print(f"\tFile {fout} exists, skipping.", flush=True)
                pass
            else:
                print(f"\tWriting clump data to file {fout}.", flush=True)
                coords_com = np.hstack((data_coms['x'][i_clump],
                                        data_coms['y'][i_clump],
                                        data_coms['z'][i_clump]))
                idx_com = data_coms['idx'][i_clump]
                R_H = coords_com_rankedf[_fin_ranked]['R_Hs'][i_clump]
                idxs_clump = coords_com_rankedf[_fin_ranked]['idxs_clumps'][i_clump]
                with open(fout, 'w') as f:
                    f.write(f"idx_com = {idx_com}\ncoords_com = {coords_com}\nR_H = {R_H}\nidxs_clump =\n")
                    np.savetxt(f, idxs_clump, fmt='%i')
                    

# ITERATIVE FUNCTIONS
def iteratively_load_data(paths_psliceout):
    """
    Iteratively loads data from psliceout files which are given by paths_psliceout.
    In:
        > paths_psliceout - (dct) paths to psliceout files
    Out:
        > data_all - (dct) dictionary containing all data
    """
    # Report function name
    print(report_function_name(), flush=True)

    # Load data
    print("\tLoading data", flush=True)
    data_all = {}
    for key in paths_psliceout:
        for path in paths_psliceout[key]:
            sub_key = path.split("/")[-1].strip(".dat")
            try:
                data_all[sub_key]
                print(f"\t\tData already read for {path}. Skipping.", flush=True)
                continue
            except KeyError:
                print(f"\t\tLoading file {path}", flush=True)
                data_all[sub_key] = load_psliceout(path)
    return data_all


def iteratively_rank_neighbors(data_all, paths_psliceout, path_ranked, n_neighbors):
    """
    Iteratively ranks neighbors for all psliceout files
    In:
        > data_all - (dct) dictionary containing all data
        > paths_psliceout - (dct) paths to psliceout files
        > path_ranked - (str) path to ranked file directory
        > n_neighbors - (int) number of nearest neighbors to find
    Out:
        > fout_ranked - (dct) dictionary containing paths to ranked files
    """
    # Report function name
    print(report_function_name(), flush=True)

    # Rank particles
    print("\tRanking particles", flush=True)
    fout_ranked = {}
    for key in paths_psliceout:
        fout_ranked[key] = []
        for path in paths_psliceout[key]:
            print(f"\t\tRanking file {path}", flush=True)
            fstring = path.split("/")[-1].strip(".dat")
            fout_ranked[key].append(rf.rank_neighbors(data_all[fstring], path, path_ranked, n_neighbors))
    return fout_ranked


def iteratively_delimit_clumps(config, fout_ranked, data_all):
    """
    Function to iteratively delimit clumps in all selected files using the available routines
    In:
        > config - (dct) dictionary containing config parameters
        > fout_ranked - (dct) dictionary containing paths to ranked files (see iteratively_rank_neighbors)
        > data_all - (dct) dictionary containing all data (see iteratively_load_data)
    Out:
        > coords_com_rankedf - (dct) dictionary containing COM coordinates, indices of members, and Hill radii of clumps for all ranked files
    """
    # Report function name
    print(report_function_name(), flush=True)

    # Use ranked data to get COM coordinates
    print("\tUsing ranked data to get COM coordinates", flush=True)

    # Calculate Roche density as threshold
    G = config["const_G"]
    Omega_K = config["const_Omega_K"]
    H_g = config["const_H_g"]
    threshold_rho = cf.calculate_density_Roche(G, Omega_K, H_g)

    # Get COM coordinates for all ranked files
    # Setting up shell parameters
    dr = config["const_dr"]
    n_shells = config["const_N_shells"]
    threshold_radius = config["const_thresh_rad"]

    # Do for all ranked files
    data_rankedf = {} # used just to check that duplicate analyses don't occur (due to sharing of ranked files between the figures)
    coords_com_rankedf = {}
    for folder in fout_ranked:
        for fin_ranked in fout_ranked[folder]:
            _fin_ranked = fin_ranked.split("/")[-1].strip(".dat")
            try:
                data_rankedf[_fin_ranked]
                print(f"\t\tFile {_fin_ranked} already processed, skipping.", flush=True)
            except KeyError:
                print(f"\t\tProcessing file {_fin_ranked}", flush=True)

                # Consolidate raw data with ranked data
                data_key = re.search("ranked_[0-9]+neigh_(.+).dat", fin_ranked).group(1)
                raw_coords, raw_idx = data_all[data_key]['raw_coords'], data_all[data_key]['raw_idx']

                # Get ranked data
                _data_ranked_red, _data_ranked = rf.read_ranked(fin_ranked, threshold_rho)

                # Get COM coordinates
                _coords_com_ranked, idxs_clumps, R_Hs_final = rf.get_com_coords(_data_ranked_red, dr, n_shells, raw_coords, raw_idx, threshold_radius, var_shell=False)

                # Remove the final "clump" since it would have been empty
                _coords_com_ranked = {key: _coords_com_ranked[key][:-1] for key in _coords_com_ranked}

                # Save data
                data_rankedf[_fin_ranked] = _data_ranked_red
                coords_com_rankedf[_fin_ranked] = {"data_com": _coords_com_ranked, "idxs_clumps": idxs_clumps, "R_Hs": R_Hs_final}
    return coords_com_rankedf


def iteratively_plot_figures(selected_figs, paths_psliceout, n_neighbors, data_all, coords_com_rankedf):
    """
    Iteratively plots figures
    """
    # Report function name
    print(report_function_name(), flush=True)
    
    # Plot figures
    print("\tPlotting figures", flush=True)
    for fig in selected_figs:
        for fin in paths_psliceout[fig]:
            _fin = fin.split("/")[-1].strip(".dat")
            _rfin = f"ranked_{n_neighbors}neigh_{_fin}"
            plot_inst = pf.PlotDustSurfdensEff(_fin, data_all[_fin], coords_com_rankedf[_rfin]['data_com'], coords_com_rankedf[_rfin]['R_Hs'], n_neighbors)
            plot_inst.plot_surfdens(save=True, show=False)


# def iteratively_plot_figures_2():
#     fig_details = pd.read_csv("fig_details.csv", sep=',', header=0)


# HELPER FUNCS
def check_paths(config):
    """
    Checks paths specified in config exist & creates them if not
    In:
        > config - (dct) dictionary containing config parameters
    """
    # Report function name
    print(report_function_name(), flush=True)
    
    # Check that paths exist, if not, create them
    for key in config:
        if key.startswith("path_"):
            if not os.path.exists(config[key]):
                print(f"\t\tCreating directory {config[key]}", flush=True)
                os.makedirs(config[key])
    return None


def report_function_name():
    """
    Reports the name of the function that called this function
    Out:
        > string - (str) name of function that called
    """
    name_func = inspect.stack()[1][3]
    name_module = inspect.stack()[1][1].split("/")[-1].strip(".py")
    string = f"({name_module}.{name_func})"
    return string


# UNITS CLASS
class units:
    def __init__(self, dust_gas_ratio, npart, code_box_size):
        """
        Class for unit conversion
        In:
            > dust_gas_ratio - (flt) dust-to-gas ratio
            > npart - (int) number of particles
            > code_box_size - (flt) size of simulation box in code units
        """
        # Mass conversion factors
        self.m_gas_tot = 3.2e31 # [g]
        self.dust_gas_ratio = dust_gas_ratio
        self.npart = npart

        # Length conversion factors
        self.physical_box_size = 50.0 # [AU]
        self.box_size = code_box_size


    def convert_to_physical_units(self, value, which='mass_d'):
        """
        Converts from code units to physical units
        In:
            > value - (flt) value to convert. If unit is 'length', this will return the physical length in AU. If unit is either 'mass_d' or 'mass_g', 'value' must be the number of particles, and this function will return the physical mass in grams.
            > which - (str) unit to convert to from either 'length', 'mass_d', or 'mass_g'
        """
        units = {
            'length': self.physical_box_size / self.box_size,
            'mass_d': self.m_gas_tot * self.dust_gas_ratio / self.npart,
            'mass_g': self.m_gas_tot * (1-self.dust_gas_ratio) / self.npart
            }
        return value*units[which]