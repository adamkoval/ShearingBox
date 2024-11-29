# %%
import os

import numpy as np
import pandas as pd
import vispy.scene

from vispy.scene import visuals

import func as f

# %%
# Read in data
# fin_raw = "../../data/Fig2_Fig4/pslice15_ml.dat"
fin_raw = "../../data/Fig7/pslice0_dm7e-3_St05-5.dat"
raw = pd.read_csv(fin_raw, sep='\s+', index_col=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'size'], header=None)
# raw = raw.sort_index()
raw_idx = raw.index.values
raw_coords = raw[['x', 'y', 'z']].values
raw_vels = raw[['vx', 'vy', 'vz']].values
raw_sizes = raw[['size']].values

# %%
# Find all the clump data files for this raw data file
com_data_dir = "../../output/com_data/"
fstr = fin_raw.split("/")[-1].split(".")[0]
fins_com_data = [com_data_dir + f for f in os.listdir(com_data_dir) if fstr in f]

# %%
# Setup
print("Initialising plot")
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers()
# vispy.app.run()

# Data prep
# Initialise pos_raw and pos_clump
pos_raw = np.column_stack([raw_coords[:, 0], raw_coords[:, 1], raw_coords[:, 2]])
pos_com = np.empty((0, 3))
pos_clump = np.empty((0, 3))
raw_idx_red = raw_idx

# # Enter main loop over clumps
for fin_com_data in fins_com_data:
    # Identify particle idxs in clump
    # fin_com_data = fins_com_data[0]
    print(f"Reading in {fin_com_data}")
    com_data = f.read_com_data(fin_com_data)
    clump_idx = com_data['idxs_clump']

    # Update the COM coordinates
    pos_com = np.vstack((pos_com, com_data['coords_com']))

    # Create a boolean mask for clump_indices and remove from raw_idx_red
    clump_mask = np.isin(raw_idx_red, clump_idx)
    raw_idx_red = raw_idx_red[~clump_mask]

    # Append clump to pos_clump and remove from pos_raw
    pos_clump = np.vstack((pos_clump, pos_raw[clump_mask]))
    pos_raw = pos_raw[~clump_mask]

# %%
    # pos_clump = pos_raw[clump_mask]
    # pos_raw = pos_raw[~clump_mask]
    # pos_com = np.column_stack([com_data['coords_com'][0], com_data['coords_com'][1], com_data['coords_com'][2]])

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
