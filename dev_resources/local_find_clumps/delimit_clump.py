# %%
import numpy as np
import matplotlib.pyplot as plt

import func as f

# %%
# Read in ranked list and get densest particle
fin_ranked = "../local_rank_neighbours/output/ranked_1000_neigh.dat"
ranked, com_coords = f.get_com_coords(fin_ranked)

# %%
# Read in raw data
fin_raw = "../psliceout1.dat"
raw_coords, raw_vels, raw_sizes, raw_idx = f.read_psliceout(fin_raw)

# %%
# Sort the distances
dr = 1e-5
n_shells = 4001
clump_rad, clump_dens, numpart, distances, sorted_distances = f.sort_distances(dr, n_shells, com_coords, raw_coords, raw_idx, var_shell=False)

# %%
# Plot analysis multiplot
plot = f.analysis_plot(clump_rad, clump_dens, numpart, sorted_distances)
plot.create()

# %%
# Plot Hill-sphere radii of shells from centre of clump outwards
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
fig, ax = plt.subplots()
ax.plot(clump_rad, R_Hs)
ax.set_xlabel("Clump radius [c.u.]")
ax.set_ylabel("Hill radius [c.u.]")
# %%
