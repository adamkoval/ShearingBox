# General functions for data loading and processing
import pandas as pd
import glob
import json
import os

import ranked_funcs as rf


def load_paths_all(path_paths):
    """
    Loads all path data from .json file
    In:
        > path_paths - (str) path to paths.json file
    Out:
        > dict_paths - (dct) dictionary containing all paths
    """
    # Load paths
    print("(gen_funcs.load_paths_all) Reading paths", flush=True)
    with open(path_paths, "r") as f:
        dict_paths = json.load(f)
    
    # Check that paths exist, if not, create them
    for key in dict_paths:
        if not os.path.exists(dict_paths[key]):
            print(f"(gen_funcs.load_paths_all) Creating directory {dict_paths[key]}", flush=True)
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
        paths_psliceout[fig] = [i for i in glob.glob(path_data + fig + "/*")]
    return paths_psliceout


def iteratively_load_data(paths_psliceout):
    """
    Iteratively loads data from psliceout files which are given by paths_psliceout.
    In:
        > paths_psliceout - (dct) paths to psliceout files
    Out:
        > data_all - (dct) dictionary containing all data
    """
    print("(gen_funcs.iteratively_load_data) Loading data", flush=True)
    data_all = {}
    for key in paths_psliceout:
        for path in paths_psliceout[key]:
            print(f"(gen_funcs.iteratively_load_data) Checking if data has already been read into memory", flush=True)
            sub_key = path.split("/")[-1].strip(".dat")
            try:
                data_all[sub_key]
                print(f"(gen_funcs.iteratively_load_data) Data already read for {path}. Skipping.", flush=True)
                continue
            except KeyError:
                print(f"(gen_funcs.iteratively_load_data) Loading file {path}", flush=True)
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
    raw = pd.read_csv(path_psliceout, sep='\s+', index_col=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'size'], header=None)
    raw_coords = raw[['x', 'y', 'z']].values
    raw_vels = raw[['vx', 'vy', 'vz']].values
    raw_sizes = raw[['size']].values
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
    print("(gen_funcs.iteratively_rank_neighbors) Ranking particles", flush=True)
    fout_ranked = {}
    for key in paths_psliceout:
        fout_ranked[key] = []
        for path in paths_psliceout[key]:
            print(f"Ranking file {path}", flush=True)
            fout_ranked[key].append(rf.rank_neighbors(data_all, path, path_ranked, n_neighbors))
    return fout_ranked