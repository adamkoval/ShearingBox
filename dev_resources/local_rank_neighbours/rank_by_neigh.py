#!/home/akoval/miniconda3/envs/python3.11_conda_env/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from time import time

# pd.set_option("display.precision", 16)

file = "../psliceout1.dat"
fstring = file.split("/")[-1].strip(".dat")

print("Reading & sorting data:", flush=True)
start = time()
data = pd.read_csv(file, sep='\s+', index_col=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'size'], header=None).sort_index()
print(f"\t\t{time() - start} s.", flush=True)

print("Starting analysis:", flush=True)
start = time()
n_neighbors = 1000  # number of nearest neighbors to find
particles = data[["x", "y", "z"]].to_numpy()
print(f"\t\t{time() - start} s.", flush=True)

# Construct the k-d tree
print("Constructing k-d tree:", flush=True)
start = time()
tree = cKDTree(particles)
print(f"\t\t{time() - start} s.", flush=True)

print("Querying k-d tree:", flush=True)
start = time()
distances, indices = tree.query(particles, k=n_neighbors+1, workers=-1)  # +1 because the particle itself is included
print(f"\t\t{time() - start} s.", flush=True)

# Rank the particles by radius
print("Ranking radii:", flush=True)
start = time()
# The distances array contains distances to the nearest neighbors for each particle
# The radius r for each particle is the distance to the n-th nearest neighbor (excluding the particle itself)
radii = distances[:, -1]  # take the distance to the n-th nearest neighbor
ranked_indices = np.argsort(radii)
print(f"\t\t{time() - start} s.", flush=True)

# Write to file
print("Writing to file:", flush=True)
start = time()
with open(f"output/ranked_{n_neighbors}neigh_{fstring}.dat", 'w+') as f:
    f.write(f"{"idx":10}\t{"radius":10}\t{"x":10}\t{"y":10}\t{"z":10}\n")
    [f.write(f"{idx+1:<10d}\t{radii[idx]:<10g}\t{particles[idx][0]:<10g}\t{particles[idx][1]:<10g}\t{particles[idx][2]:<10g}\n") for idx in ranked_indices]
print(f"\t\t{time() - start} s.", flush=True)