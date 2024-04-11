#! /bin/bash
# source /shared_new/ptoulme/axlearn/venv/bin/activate
source /shared/apoorvgu/jax-21/bin/activate
source ./setup.sh
source ./train_setup.sh

echo "==============================================="
apt list | grep neuron
pip freeze | grep neuron
echo "==============================================="

rm -rf /shared/apoorvgu/axlearn/axlearn/common/neuron_tests/test/*
rm -rf /shared/apoorvgu/axlearn/axlearn/common/neuron_tests/compiler_dump
rm -rf /shared/apoorvgu/axlearn/axlearn/common/neuron_tests/jax_dump
rm -rf /shared/apoorvgu/axlearn/axlearn/common/neuron_tests/jax4_dump
OUTPUT_DIR=/shared/apoorvgu/axlearn/axlearn/common/neuron_tests/test/
# DATA_DIR=FAKE
DATA_DIR=gs://axlearn-public/tensorflow_datasets
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-test \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR --jax_backend=neuron