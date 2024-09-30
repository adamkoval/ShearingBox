# %%
import pandas as pd
import numpy as np
import vispy.scene
from vispy.scene import visuals

import clump_utils as clu
# import cyl_utils as cyu
import plot_utils as pu

# %%
# Read in data
fin_raw = "../psliceout1.dat"
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
# Setup
print("Initialising plot")
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers()
# vispy.app.run()

# %%
# Data prep
pos_raw = np.column_stack([raw_coords[:,0], raw_coords[:,1], raw_coords[:,2]])
pos_com = np.column_stack([com_coords[0], com_coords[1], com_coords[2]])

# Adding to plot
print("Adding data to scatter")
# pos = np.vstack((pos_all_sub, pos_clump))
# colors = np.vstack((col_all_sub, col_clump))
scatter.set_data(pos_com, edge_width=0, face_color='r', size=1000, symbol='o')
scatter.set_data(pos_raw, edge_width=0, face_color=(0.8,0.7,0.7), size=1, symbol='o')

# Plotting
print("Adding view to scatter and finishing up")
view.add(scatter)
view.camera = 'turntable'
axis = visuals.XYZAxis(parent=view.scene)
import sys
if sys.flags.interactive != 1:
    vispy.app.run()