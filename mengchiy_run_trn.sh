#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --time=00:60:00

srun  --kill-on-bad-exit=1  run_step_time_multinodes_test.sh fuji-8B-v2