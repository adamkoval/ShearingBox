# Functions for ranking particles based on their distance to the n-th nearest neighbor
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

from scipy.spatial import cKDTree
from time import time

import gen_funcs as gf
import clump_funcs as cf


def read_ranked(fin_ranked, threshold_rho):
    """
    Reads in ranked file and returns data
    In:
        > fin_ranked - (str) path to ranked file
        > threshold_rho - (flt) threshold density to reduce search space
    Out:
        > ranked_red - (df) reduced ranked data
        > ranked - (df) ranked data
    """
    # Report function name
    print(gf.report_function_name(), flush=True)

    # Access ranked
    print(f"\tReading in and reducing ranked file {fin_ranked}", flush=True)
    ranked = pd.read_csv(fin_ranked, sep='\t', skiprows=1, names=["idx", "radius", "x", "y", "z"])

    # Calculate threshold radius based on threshold density to reduce the search space
    n_neighbors = int(re.search("([0-9]+)neigh", fin_ranked).group(1))
    threshold_radius = (n_neighbors / threshold_rho * 3/4 * 1/np.pi)**1/3

    # Access ranked and mask particles based on threshold radius
    ranked = pd.read_csv(fin_ranked, sep='\t', skiprows=1, names=["idx", "radius", "x", "y", "z"])
    ranked_red = ranked[ranked['radius'] < threshold_radius]
    return ranked_red, ranked


def get_com_coords(data_ranked, dr, n_shells, raw_coords, raw_idx, threshold_radius, var_shell=False):
    """
    Reads in ranked file and returns coordinates of centre-of-mass of clumps according to ranked
    In:
        > data_ranked - (df) ranked data
        > dr - (flt) shell width (if var_shell=True then this sets the first shell width)
        > n_shells - (int) number of shells
        > raw_coords - (NParr) xyz coordinates of all particles in slice
        > raw_idx - (NParr) indices of all particles in slice
        > threshold_radius - (flt) threshold radius to reduce clump analysis to (NOT TO BE CONFUSED WITH threshold_radius IN rank_neighbors)
        > var_shell - (bool) whether to vary shell width
    Out:
        > coords_com_all - (NParr) xyz coordinates of centre-of-mass of clump (currently just for one COM)
        > clump_indices - (NParr) indices of particles in clump
    """
    # Report function name
    print(gf.report_function_name(), flush=True)

    # # Prepare the data for the iterative clump analysis
    raw_coords_rem = raw_coords
    raw_idx_rem = raw_idx
    data_ranked_rem = data_ranked

    # Iteratively find the centres of mass of disconnected clumps
    print(f"\tFinding centres of mass of disconnected clumps (threshold radius:  {threshold_radius})", flush=True)
    coords_com_all = {key: [] for key in data_ranked_rem.keys()}
    idxs_clump = []
    R_Hs_final = []
    g = 0
    while g<20:
        # Add the densest particle to list of COM coordinates
        try:
            coords_com_curr = data_ranked_rem[['x', 'y', 'z']].values[0]
        except IndexError:
            print(f"\t\t\tNo COM satisfying reduction condition. Exiting.", flush=True)
            return np.empty(1), np.empty(1), np.empty(1)
        [coords_com_all[key].append(data_ranked_rem[key].values[0]) for key in coords_com_all.keys()]
        print(f"\n\t\t\tCOM of clump {g}: {coords_com_curr}", flush=True)

        # Reduce the search space to a sphere of threshold_radius around the current densest clump
        distances_to_com = np.sqrt((raw_coords_rem[:, 0] - coords_com_curr[0])**2 + (raw_coords_rem[:, 1] - coords_com_curr[1])**2 + (raw_coords_rem[:, 2] - coords_com_curr[2])**2)
        print(f"\t\t\tNumber of particles in raw search space: {len(raw_coords_rem)}", flush=True)
        print(f"\t\t\tMinimum distance to COM: {np.min(distances_to_com):.4e}", flush=True)
        raw_coords_red = raw_coords_rem[distances_to_com < threshold_radius]
        raw_idx_red = raw_idx_rem[distances_to_com < threshold_radius]
        print(f"\t\t\tNumber of particles in reduced search space: {len(raw_coords_red)}", flush=True)
        if len(raw_coords_red) == 0:
            print(f"\t\t\tNo more particles in search space. Exiting.", flush=True)
            break
        
        # Now find the particles in the clump
        clump_rad, clump_dens, numpart, distances, sorted_distances = cf.sort_distances(dr, n_shells, coords_com_curr, raw_coords_red, raw_idx_red, var_shell=False)
        print(f"\t\t\tNumber of particles in preclump: {int(numpart[-1])}", flush=True)
        
        # Get the Hill-sphere radii of shells from centre of clump outwards
        Omega = 1
        G = 1
        R_Hs = np.array([cf.R_H(part * gf.m_swarm, Omega, G) for part in numpart])

        # Find first shell where clump radius exceeds Hill radius
        mask = clump_rad > R_Hs
        indices = np.where(mask)[0]
        if indices.size > 0:
            shelli_cross = indices[0]
        print(f"\t\t\tCrossing point: {shelli_cross}", flush=True)
        print(f"\t\t\tHill Radius: {R_Hs[shelli_cross]}", flush=True)
        R_Hs_final.append(R_Hs[shelli_cross])
        
        # Identify particle idxs in clump
        sorted_indices = np.argsort(distances)
        clump_indices = raw_idx_red[sorted_indices[:int(numpart[shelli_cross])]]
        print(f"\t\t\tNumber of particles in postclump: {numpart[shelli_cross]}", flush=True)
        if numpart[-1] == numpart[shelli_cross]:
            print(f"WARNING: All particles in search space are in clump. Adjust your shell parameters. Exiting.", flush=True)
            break

        # Remove the particles in the clump from the ranked data and the original unreduced search space
        data_ranked_rem = data_ranked_rem[~data_ranked_rem['idx'].isin(clump_indices)]
        print(f"\t\t\tNumber of particles in ranked data at the end: {len(data_ranked_rem)}", flush=True)
        raw_coords_rem = raw_coords_rem[~np.isin(raw_idx_rem, clump_indices)]
        raw_idx_rem = raw_idx_rem[~np.isin(raw_idx_rem, clump_indices)]

        # Append the indices of the clump to the list
        idxs_clump.append(clump_indices)
        g += 1
    return coords_com_all, idxs_clump, R_Hs_final


def rank_neighbors(data, path_psliceout, path_ranked, n_neighbors):
    """
    Reads in PENCIL slice data file and writes a ranked file based on the n-th nearest neighbor
    In:
        > path_psliceout - (str) path to psliceout file
        > path_ranked - (str) path to ranked directory
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
    particles = data['raw_coords']
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