#!/usr/bin/env bash
sudo dpkg -i /home/ptoulme/compiler_5th/aws-neuronx-runtime-lib-2.x.x.x-b0ad3b9d9.deb
sudo dpkg -i /home/ptoulme/compiler_5th/aws-neuronx-collectives-2.x.x.x-99c2ee88a.deb

PY_VENV_PATH="/home/ptoulme/axlearn_venv/bin/activate"
source ${PY_VENV_PATH}

cd /axlearn

ARTIFACTS_PATH="/home/ptoulme/artifacts"
TIMESTAMP=$(date +"%y%m%d%H%M%S")
export TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${TIMESTAMP}"
mkdir -p "$TEST_ARTIFACTS_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/flags.sh"

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# Run the training script
python -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v1 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=trn2 

