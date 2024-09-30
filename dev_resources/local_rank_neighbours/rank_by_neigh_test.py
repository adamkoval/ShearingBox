#!/home/akoval/miniconda3/envs/python3.11_conda_env/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from time import time

# pd.set_option("display.precision", 16)

file = "../psliceout3;.dat"
fstring = file.split("/")[-1].strip(".dat")

print("Reading & sorting data.", flush=True)
start = time()
data = pd.read_csv(file, sep='\s+', index_col=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'size'], header=None).sort_index()
print(f"Time spent reading & sorting: {time() - start} s.", flush=True)

print("Starting analysis loop.", flush=True)
n_neighbors = 1000  # number of nearest neighbors to find
subsets = [10**j for j in range(1, 8)]
times = []
for i, subset in enumerate(subsets):
    print(f"Working on subset {subset}.", flush=True)
    times.append(time())
    particles = data[["x", "y", "z"]].iloc[:subset].to_numpy()

    # Construct the k-d tree
    tree = cKDTree(particles)

    distances, indices = tree.query(particles, k=n_neighbors+1)  # +1 because the particle itself is included
    times[i] = time() - times[i]
    print(f"Done working on {subset}.", flush=True)

    if subset == subsets[-1]:
        print(f"Writing results for {subset} parts.", flush=True)
        # The distances array contains distances to the nearest neighbors for each particle
        # The radius r for each particle is the distance to the n-th nearest neighbor (excluding the particle itself)
        radii = distances[:, -1]  # take the distance to the n-th nearest neighbor

        # Rank the particles by radius
        ranked_indices = np.argsort(radii)

        # Write to file
        with open(f"output/ranked_{n_neighbors}neigh_{fstring}.dat", 'w+') as f:
            f.write(f"{"idx":10}\t{"radius":10}\t{"x":10}\t{"y":10}\t{"z":10}\n")
            [f.write(f"{idx+1:<10d}\t{radii[idx]:<10g}\t{particles[idx][0]:<10g}\t{particles[idx][1]:<10g}\t{particles[idx][2]:<10g}\n") for idx in ranked_indices]

        # Plot times to completion
        fig, ax = plt.subplots()
        ax.plot(subsets, times, ls='-', marker='*', c='b')
        ax.set_xlabel("N-particles")
        ax.set_ylabel("Time to search completion (CPU seconds)")
        ax.set_title(f"k-d tree search scaling for {n_neighbors} neighbours")
        plt.savefig(f"output/kDtree_scaling_{n_neighbors}neigh_{fstring}.png", format='png')
        plt.close()