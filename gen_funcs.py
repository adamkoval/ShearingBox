import pandas as pd
import glob


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