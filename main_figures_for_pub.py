# MAIN SCRIPT FOR GENERATING FIGURES FOR PUBLICATION
# %%
import gen_funcs as gf
import ranked_funcs as rf
import clump_funcs as cf
import os

# %%
# Select desired figures
print("Running setup operations", flush=True)
which_figs = "selected" # "all" or "selected"
all_figs = ["Fig2_Fig4", "Fig6", "Fig7", "Fig8"]
selected_figs = ["Fig2_Fig4"]

if which_figs == "all":
    figs = all_figs
elif which_figs == "selected":
    figs = selected_figs

# %%
# Load paths
path_paths = "paths.json"
dict_paths = gf.load_paths_all(path_paths)

# %%
# Load data
path_data = dict_paths["path_data"]
paths_psliceout = gf.load_paths_psliceout(path_data, figs)
data_all = gf.iteratively_load_data(paths_psliceout)

# %%
# Rank particles
path_ranked = dict_paths["path_ranked"]
n_neighbors = 1000
fout_ranked = gf.iteratively_rank_neighbors(data_all, paths_psliceout, path_ranked, n_neighbors)

# # %%
# # Use ranked data to get COM coordinates
# print("Using ranked data to get COM coordinates", flush=True)
# data_ranked, coords_com_ranked = rf.get_com_coords(dict_paths["path_ranked"])