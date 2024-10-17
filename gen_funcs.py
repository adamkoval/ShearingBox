# General functions for data loading and processing
import pandas as pd
import glob
import inspect
import json
import os

import ranked_funcs as rf

m_swarm = 1.44e-5  # particle mass [code units]

def load_paths_all(path_paths):
    """
    Loads all path data from .json file
    In:
        > path_paths - (str) path to paths.json file
    Out:
        > dict_paths - (dct) dictionary containing all paths
    """
    # Report function name
    print(report_function_name(), flush=True)

    # Load paths
    print("\tReading paths", flush=True)
    with open(path_paths, "r") as f:
        dict_paths = json.load(f)
    
    # Check that paths exist, if not, create them
    for key in dict_paths:
        if not os.path.exists(dict_paths[key]):
            print(f"\t\tCreating directory {dict_paths[key]}", flush=True)
            os.makedirs(dict_paths[key])
    return dict_paths


def load_paths_psliceout(path_data, selected_figs):
    """
    Loads paths to psliceout files
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
            fout_ranked[key].append(rf.rank_neighbors(data_all, path, path_ranked, n_neighbors))
    return fout_ranked


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