#! /bin/bash
source /shared/apoorvgu/jax-21/bin/activate
source ./setup.sh
source ./train_setup.sh

OUTPUT_DIR=./c4_test_dump
DATA_DIR=gs://axlearn-public/tensorflow_datasets
# python3 -m axlearn.common.launch_trainer_main \
#     --module=text.gpt.c4_trainer --config=fuji-test \
#     --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR --jax_backend=neuron

python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-test \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=neuron-trn1.32xlarge-64 \
    --distributed_coordinator=$MASTER_ADDR:$MASTER_PORT --num_processes=$SLURM_NTASKS \
    --process_id=$SLURM_NODEID