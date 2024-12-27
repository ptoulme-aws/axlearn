#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --exclusive
#SBATCH --nodes=1


#Command : run_trn.sh <launch_script> <launch script options> 
#NOTE :  <launch script options> are defined by <launch_script>
#Example :
# run_trn.sh run_trainer.sh fs_main2 my_test setup7b.l32b64.n4.sh
srun  --kill-on-bad-exit=1 "$@"

