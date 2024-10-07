# %%
import pandas as pd
import numpy as np
import vispy.scene
from vispy.scene import visuals

import func as f

# %%
# Read in data
fin_raw = "../../data/old/psliceout1.dat"
raw = pd.read_csv(fin_raw, sep='\s+', index_col=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'size'], header=None)
raw_coords = raw[['x', 'y', 'z']].values
raw_vels = raw[['vx', 'vy', 'vz']].values
raw_sizes = raw[['size']].values
raw_idx = raw.index.values

# %%
fin_ranked = "../local_rank_neighbours/output/ranked_1000neigh_psliceout1.dat"
ranked = pd.read_csv(fin_ranked, sep='\t', skiprows=1, names=["idx", "radius", "x", "y", "z"])
com = ranked.iloc[0]
com_coords = com[['x', 'y', 'z']].values

# %%
# Sort the distances
dr = 3e-4
n_shells = 5100
clump_rad, clump_dens, numpart, distances, sorted_distances = f.sort_distances(dr, n_shells, com_coords, raw_coords, raw_idx, var_shell=False)

# %%
# Get Hill-sphere radii of shells from centre of clump outwards
Sigma_g = 1
G = 1
Omega_K = 1
H_g = np.pi
Sigma_R = np.sqrt(2*np.pi) * 3.5 * Omega_K**2 * H_g / G
print(f"Sigma_R = {Sigma_R:.4f}")
# rho_m0 = Sigma_g / 2 / H_g
# R_Hs = f.find_R_Hs(numpart, rho_m0)
Omega = 1
G = 1
R_Hs = f.find_R_Hs(numpart, Omega, G)

# %%
# Find first shell where clump radius exceeds Hill radius
mask = clump_rad > R_Hs
indices = np.where(mask)[0]
if indices.size > 0:
    idx_cross = indices[0]

# %%
# Identify particle idxs in clump
sorted_indices = np.argsort(distances)
clump_indices = raw_idx[sorted_indices[:int(numpart[idx_cross])]]

# %%
# Data prep
# Create a boolean mask for clump_indices
clump_mask = np.zeros(len(raw_coords), dtype=bool)
clump_mask[sorted_indices[:int(numpart[idx_cross])]] = True

# Separate pos_raw into pos_clump and the remaining particles
pos_raw = np.column_stack([raw_coords[:, 0], raw_coords[:, 1], raw_coords[:, 2]])
pos_clump = pos_raw[clump_mask]
pos_raw = pos_raw[~clump_mask]

# Add the COM coordinates to the plot
# pos_raw = np.column_stack([raw_coords[:,0], raw_coords[:,1], raw_coords[:,2]])
pos_com = np.column_stack([com_coords[0], com_coords[1], com_coords[2]])

# %%
# Setup
print("Initialising plot")
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers()
# vispy.app.run()

# Adding to plot
print("Adding data to scatter")
# pos = np.vstack((pos_all_sub, pos_clump))
# colors = np.vstack((col_all_sub, col_clump))
# scatter.set_data(pos_com, edge_width=0, face_color='r', size=1000, symbol='o')
# scatter.set_data(pos_raw, edge_width=0, face_color=(0.8,0.7,0.7), size=1, symbol='o')
col_clump = np.array([[1, 0, 0]] * len(pos_clump))
col_raw = np.array([[1, 1, 1]] * len(pos_raw))
col_com = np.array([[0, 1, 0]] * len(pos_com))
col_all = np.vstack((col_raw, col_clump, col_com))
pos_all = np.vstack((pos_raw, pos_clump, pos_com))
size_clump = np.array([1] * len(pos_clump))
size_raw = np.array([.5] * len(pos_raw))
size_com = np.array([10] * len(pos_com))
size_all = np.hstack((size_raw, size_clump, size_com))
scatter.set_data(pos_all, edge_width=0, face_color=col_all, size=size_all, symbol='o')

# scatter.set_data(pos_com, edge_width=0, face_color='r', size=1000, symbol='o')
# scatter.set_data(pos_raw, edge_width=0, face_color='b', size=5, symbol='o')
# scatter.set_data(pos_clump, edge_width=0, face_color='g', size=5, symbol='o')

# Plotting
print("Adding view to scatter and finishing up")
view.add(scatter)
view.camera = 'turntable'
axis = visuals.XYZAxis(parent=view.scene)
import sys
if sys.flags.interactive != 1:
    vispy.app.run()
# %%
