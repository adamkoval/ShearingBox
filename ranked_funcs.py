# Functions for ranking particles based on their distance to the n-th nearest neighbor
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from scipy.spatial import cKDTree
from time import time

import gen_funcs as gf


def get_com_coords(fin_ranked, n_neighbors, threshold_rho):
    """
    Reads in ranked file and returns coordinates of centre-of-mass of clumps according to ranked
    In:
        > fin_ranked - (str) path to ranked file
    Out:
        > com_coords - (arr) xyz coordinates of centre-of-mass of clump (currently just for one COM)
    """
    # Report function name
    print(gf.report_function_name(), flush=True)

    # Calculate threshold radius based on threshold density to reduce the search space
    threshold_radius = (n_neighbors / threshold_rho * 3/4 * 1/np.pi)**1/3

    # Access ranked and mask particles based on threshold radius
    ranked = pd.read_csv(fin_ranked, sep='\t', skiprows=1, names=["idx", "radius", "x", "y", "z"])
    ranked_red = ranked[ranked['radius'] < threshold_radius]

    # Find the centres of mass of disconnected clumps
    com_coords = ranked_red[['x', 'y', 'z', 'radius']].values
    # for idx in range(len(ranked)):
    #     com = ranked.iloc[idx]
    #     com_coords.append(com[['x', 'y', 'z']].values)
        # if ranked.iloc[idx]['radius'] < threshold_radius:
        #     com = ranked.iloc[0]
        #     com_coords.append(com[['x', 'y', 'z']].values)
        # else:
        #     pass
    return ranked, com_coords


def rank_neighbors(data_all, path_psliceout, path_ranked, n_neighbors):
    """
    Reads in PENCIL slice data file and writes a ranked file based on the n-th nearest neighbor
    In:
        > path_psliceout - (str) path to psliceout file
        > n_neighbors - (int) number of nearest neighbors to find
    Out:
        > fout_ranked - (str) path to ranked file
    """
    # Report function name
    print(gf.report_function_name(), flush=True)

    # Checking existing ranked file
    print("\tChecking if ranked file exists", flush=True)
    fstring = path_psliceout.split("/")[-1].strip(".dat")
    fout_ranked = f"{path_ranked}ranked_{n_neighbors}neigh_{fstring}.dat"
    if os.path.exists(fout_ranked):
        print(f"\t\tRanked file {fout_ranked} already exists. Skipping.", flush=True)
        return fout_ranked
    else:
        print(f"\t\tRanked file {fout_ranked} does not exist. Proceeding.", flush=True)
        pass

    # Access data
    print("\t\t\tAccessing  data:", flush=True)
    start = time()
    particles = data_all[fstring]['raw_coords']
    print(f"\t\t\t\t{time() - start} s.", flush=True)

    # Construct the k-d tree
    print("\t\t\tConstructing k-d tree:", flush=True)
    start = time()
    tree = cKDTree(particles)
    print(f"\t\t\t\t{time() - start} s.", flush=True)

    print("\t\t\tQuerying k-d tree:", flush=True)
    start = time()
    distances, indices = tree.query(particles, k=n_neighbors+1, workers=-1)  # +1 because the particle itself is included
    print(f"\t\t\t\t{time() - start} s.", flush=True)

    # Rank the particles by radius
    print("\t\t\tRanking radii:", flush=True)
    start = time()
    # The distances array contains distances to the nearest neighbors for each particle
    # The radius r for each particle is the distance to the n-th nearest neighbor (excluding the particle itself)
    radii = distances[:, -1]  # take the distance to the n-th nearest neighbor
    ranked_indices = np.argsort(radii)
    print(f"\t\t\t\t{time() - start} s.", flush=True)

    # Write to file
    print("\t\t\tWriting to file:", flush=True)
    start = time()
    with open(fout_ranked, 'w+') as f:
        f.write(f"{'idx':10}\t{'radius':10}\t{'x':10}\t{'y':10}\t{'z':10}\n")
        [f.write(f"{idx+1:<10d}\t{radii[idx]:<10g}\t{particles[idx][0]:<10g}\t{particles[idx][1]:<10g}\t{particles[idx][2]:<10g}\n") for idx in ranked_indices]
    print(f"\t\t\t\t{time() - start} s.", flush=True)
    return fout_ranked