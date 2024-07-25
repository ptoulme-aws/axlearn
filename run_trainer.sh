#!/usr/bin/env bash

# Editable paths

# CONDA
# CONDA_HOME="/shared/thangakr/conda"
# CONDA_ENV_NAME="tot"
# Source conda environment
# source ${CONDA_HOME}/bin/activate ${CONDA_ENV_NAME}

# VENV
PY_VENV_PATH="/shared/ptoulme/axlearn/venv/bin/activate"
source ${PY_VENV_PATH}

NEURON_DUMP_PATH=${PWD}/neuron_dump_no_artifacts
HLO_DUMP_PATH=${PWD}/hlo_dump

# Install runtime and collectives library. This is only needed in internal dev cluster
# Remove this before release
source ./bigcluster_setup.sh
# export NEURON_SCRATCHPAD_PAGE_SIZE=2048
# --hbm-scratchpad-page-size=2048
# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --distribution-strategy=llm-training"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation --verbose=INFO"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"

# --internal-hlo2tensorizer-options='--recursive-layer-det'

# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
export NEURON_FSDP=1
#export NEURON_TRANSFORMER_SHARDING=1

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1

# Neuron env vars for distributed training based on SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
num_nodes=$(echo "$nodes" | wc -l)
process_idx=$(echo "$nodes" | grep -n "$SLURMD_NODENAME" | cut -d: -f1)
devices_per_node=32
MASTER_ADDR=$(echo "$nodes" | head -n 1)
MASTER_PORT=41000
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$((process_idx - 1))
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_LOG_LEVEL="warn"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export XLA_FLAGS="--xla_force_host_platform_device_count=32 --xla_dump_hlo_as_text --xla_dump_to=./jax_dump_vlog2 --xla_dump_hlo_pass_re='.*'"
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_VLOG_LEVEL=0
OUTPUT_DIR="/shared_new/ptoulme/fsdp3"


DATA_DIR="/shared_new/ptoulme/c4_800gb/tensorflow_datasets"
# Run the training script
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v1 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=neuron-trn1.32xlarge-64 \
    --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
    --process_id=$NEURON_PJRT_PROCESS_INDEX
