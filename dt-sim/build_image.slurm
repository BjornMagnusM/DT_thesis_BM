#!/bin/bash
#SBATCH --job-name=build_sif
#SBATCH --partition=allgroups
#SBATCH --time=01:00:00
#SBATCH --mem=16G

module load singularity  # or singularity

cd $HOME/DT_thesis_BM/dt-sim

singularity build duckie_rl.sif duckie_rl.def