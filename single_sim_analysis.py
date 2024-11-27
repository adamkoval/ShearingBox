#!/home/akoval/miniconda3/envs/python3.11_conda_env/bin/python
# SCRIPT TO ANALYSE CLUMPS IN SINGLE FIGURE
import func_gen as gf
import func_ranked as rf
import func_clump as cf
import func_plot as pf

import os
import numpy as np
import re
from scipy.linalg import norm

# Setup
print("Running setup operations", flush=True)
config = gf.read_config("config.json")
gf.check_paths(config)

# Assign paths to variables
path_data = config["path_data"]
selected_figs = config["arr_figs"] # Folders to plot figures from. Edit config.json as e.g., ["Fig2_Fig4", "Fig6", "Fig7", "Fig8"] or ["old"]
paths_psliceout = gf.read_paths_psliceout(path_data, selected_figs)
path_out = config["path_out"]
path_ranked = config["path_ranked"]

# Load data
fin = "data/Fig8/pslice15_dm1e-2_St10-100.dat"
data = gf.load_psliceout(fin)

# Rank neighbours
fout_ranked = rf.rank_neighbors(data, fin, "output/ranked/", 1000)

# Find clumps
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

# Consolidate raw data with ranked data
data_key = re.search("ranked_[0-9]+neigh_(.+).dat", fout_ranked).group(1)
raw_coords, raw_idx = data['raw_coords'], data['raw_idx']

# Get ranked data
_data_ranked_red, _data_ranked = rf.read_ranked(fout_ranked, threshold_rho)

# Get COM coordinates
_coords_com_ranked, idxs_clump, R_Hs_final = rf.get_com_coords(_data_ranked_red, dr, n_shells, raw_coords, raw_idx, threshold_radius, var_shell=False)