#! /bin/bash
source /shared_new/ptoulme/axlearn/venv/bin/activate
source ./setup.sh
source ./train_setup.sh

OUTPUT_DIR=./c4_test_dump
DATA_DIR=gs://axlearn-public/tensorflow_datasets
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-test \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR --jax_backend=neuron