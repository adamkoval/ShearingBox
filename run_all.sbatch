#!/bin/bash
#SBATCH --job-name=shearing_box_analysis
#SBATCH --time=12:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G

# If bash is your shell
ulimit -s unlimited

# Run main file
./main_figures_for_pub.py > run_log_all

# Rename log
./rename_log.py run_log_all