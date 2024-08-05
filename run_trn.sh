#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 127
#SBATCH --nodes=8

srun  --kill-on-bad-exit=1  run_trainer.sh