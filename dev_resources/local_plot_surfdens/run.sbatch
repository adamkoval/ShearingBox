#!/bin/bash
#SBATCH --job-name=plot_dust_surfdens
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

# If bash is your shell
ulimit -s unlimited

./plot_dust_surfdens_eff.py -f ../psliceout1.dat > run_log