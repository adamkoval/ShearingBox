# MAIN SCRIPT FOR GENERATING FIGURES FOR PUBLICATION
# %%
import gen_funcs as gf
import ranked_funcs as rf
import clump_funcs as cf
import json
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
print("Reading paths", flush=True)
path_paths = "paths.json"
with open(path_paths, "r") as f:
    dict_paths = json.load(f)

# Check that paths exist, if not, create them
for key in dict_paths:
    if not os.path.exists(dict_paths[key]):
        print(f"Creating directory {dict_paths[key]}", flush=True)
        os.makedirs(dict_paths[key])

# %%
# Load data
print("Loading data", flush=True)
path_data = dict_paths["path_data"]
paths_psliceout = gf.load_paths_psliceout(path_data, figs)
data_all = {}
for key in paths_psliceout:
    for path in paths_psliceout[key]:
        print(f"Checking if data has already been read into dict", flush=True)
        sub_key = path.split("/")[-1].strip(".dat")
        try:
            data_all[sub_key]
            print(f"Data already read for {path}. Skipping.", flush=True)
            continue
        except KeyError:
            print(f"Loading file {path}", flush=True)
            data_all[sub_key] = gf.load_psliceout(path)

# %%
# Rank particles
print("Ranking particles", flush=True)
path_ranked = dict_paths["path_ranked"]
n_neighbors = 1000
fout_ranked = {}
for key in paths_psliceout:
    fout_ranked[key] = []
    for path in paths_psliceout[key]:
        print(f"Ranking file {path}", flush=True)
        fout_ranked[key].append(rf.rank_neighbours(data_all, path, path_ranked, n_neighbors))

# # %%
# # Use ranked data to get COM coordinates
# print("Using ranked data to get COM coordinates", flush=True)
# data_ranked, coords_com_ranked = rf.get_com_coords(dict_paths["path_ranked"])