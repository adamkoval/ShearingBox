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
fin_raw = "../../data/old/psliceout1.dat"
raw_coords, raw_vels, raw_sizes, raw_idx = f.read_psliceout(fin_raw)

# %%
# Sort the distances
dr = 3e-4
n_shells = 5100
clump_rad, clump_dens, numpart, distances, sorted_distances = f.sort_distances(dr, n_shells, com_coords, raw_coords, raw_idx, var_shell=False)

# # %%
# # Plot analysis multiplot
# plot = f.analysis_plot(clump_rad, clump_dens, numpart, sorted_distances, 2e-4)
# plot.create()

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
# Plotting
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].plot(clump_rad, R_Hs)
ax[0].set_xlabel("Clump radius [c.u.]")
ax[0].set_ylabel("Hill radius [c.u.]")
ax[0].plot(clump_rad[idx_cross], R_Hs[idx_cross], "ro", label=f"Crossing point~{R_Hs[idx_cross]:.2f}")
ax[0].axvline(clump_rad[idx_cross], color="r", linestyle="--", lw=.5)
ax[0].axhline(R_Hs[idx_cross], color="r", linestyle="--", lw=.5)
ax[0].legend()

ax[1].plot([i for i in range(len(R_Hs))], R_Hs, label="Hill radius")
ax[1].plot([i for i in range(len(R_Hs))], clump_rad, label="Clump radius")
ax[1].plot(idx_cross, R_Hs[idx_cross], "ro")
ax[1].set_xlabel("Shell number")
ax[1].set_ylabel("Radius [c.u.]")
ax[1].legend()

# Get physical mass of clump
m_gas_tot = 3.2e31 # [g]
dust_gas_ratio = 1e-2
m_swarm = 1.44e-5 # [c.u.]
clump_mass_d = numpart[idx_cross] * m_gas_tot * dust_gas_ratio / len(raw_coords)
m_earth = 5.972e27 # [g]

# Report physical mass of clump
ax[1].text(.2, .05, f"Num. part @ crossing\npoint = {int(numpart[idx_cross])}\n$\\therefore$ clump dust mass = {clump_mass_d/m_earth:.3f} M$_\\oplus$", transform=ax[1].transAxes)

plt.tight_layout()
plt.show()
# %%
