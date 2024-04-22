#! /bin/bash
set -e

# LOG_FILE="./${SLURM_JOB_ID}_${SLURM_NODEID}.log"
# exec >"$LOG_FILE" 2>&1
# echo "The name of this node is: $SLURMD_NODENAME"

source /shared/apoorvgu/jax-21/bin/activate
source ./setup.sh
source ./train_setup.sh

OUTPUT_DIR=./c4_test_dump
DATA_DIR=gs://axlearn-public/tensorflow_datasets

python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=neuron-trn1.32xlarge-64
