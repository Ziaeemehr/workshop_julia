#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=...
#SBATCH --mem-per-cpu=400M
#SBATCH --time=00:10:00

module load StdEnv/2020 julia/1.5.2
julia -t $SLURM_CPUS_PER_TASK heavyThreads.jl
