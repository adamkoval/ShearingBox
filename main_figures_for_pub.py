# MAIN SCRIPT FOR GENERATING FIGURES FOR PUBLICATION
# %%
import gen_funcs as gf
import ranked_funcs as rf
import clump_funcs as cf
import plot_funcs as pf

import os
import numpy as np
import re
from scipy.linalg import norm

# %%
# Setup
print("Running setup operations", flush=True)
config = gf.read_config("config.json")
gf.check_paths(config)

# %%
# Assign paths to variables
path_data = config["path_data"]
selected_figs = config["arr_figs"] # Folders to plot figures from. Edit config.json as e.g., ["Fig2_Fig4", "Fig6", "Fig7", "Fig8"] or ["old"]
paths_psliceout = gf.read_paths_psliceout(path_data, selected_figs)
path_out = config["path_out"]
path_ranked = config["path_ranked"]

# %%
# Load data
data_all = gf.iteratively_load_data(paths_psliceout)

# %%
# Rank particles
path_ranked = config["path_ranked"]
n_neighbors = config["const_N_neigh"]
fout_ranked = gf.iteratively_rank_neighbors(data_all, paths_psliceout, path_ranked, n_neighbors)

# %%
# Find clumps
coords_com_rankedf = gf.iteratively_delimit_clumps(config, fout_ranked, data_all)

# %%
# Plotting
gf.iteratively_plot_figures(selected_figs, paths_psliceout, n_neighbors, data_all, coords_com_rankedf)