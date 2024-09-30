# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit

import func as f

m_part = 1.44e-5 # particle mass [code units]

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
clump_rad, clump_dens, numpart, distances, sorted_distances = f.sort_distances(dr, n_shells, com_coords, raw_coords, raw_idx)

# %%
xlim = 2e-4
fig, ax = plt.subplots(1,3, figsize=(10,5))

ax[0].plot(clump_rad, clump_dens)
ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0].set_xlabel("Radius [c.u.]")
ax[0].set_ylabel("Density [c.u.]")
ax[0].set_xlim(0, xlim)

ax[1].plot(clump_rad[clump_rad < xlim], numpart[clump_rad < xlim])
ax[1].set_xlabel("Radius [c.u.]")
ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1].set_ylabel("Num part")
ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
lim_numpart = np.max(numpart[clump_rad < xlim])
ax[1].axhline(lim_numpart, c='r', label=f"npart = {lim_numpart}")
# ax[1].text(f"{lim_numpart}", 0.0, lim_numpart)
ax[1].set_xlim(0, xlim)
ax[1].legend()

@jit(nopython=True)
def mean_dists(sorted_distances):
    cumulative_sums = np.cumsum(sorted_distances)
    means = cumulative_sums / np.arange(1, len(sorted_distances) + 1)
    return means

ax[2].plot(mean_dists(np.array(sorted_distances[:int(lim_numpart)])), range(len(sorted_distances[:int(lim_numpart)])))
ax[2].set_xlabel("Mean distance (c.u.)")
ax[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2].set_ylabel("Num part")
ax[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.show()
# %%
