#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 127
#SBATCH --exclusive
#SBATCH --nodes=1

srun  --kill-on-bad-exit=1  run_test.sh