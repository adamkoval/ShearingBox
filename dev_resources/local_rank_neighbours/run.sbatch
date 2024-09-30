#!/bin/bash
#SBATCH --job-name=shearing_box_analysis
#SBATCH --time=12:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G

# If bash is your shell
ulimit -s unlimited

./rank_by_neigh.py > run_log