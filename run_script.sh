#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --job-name=ag_test
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=01:30:00
srun  --kill-on-bad-exit=1  run_mainline.sh