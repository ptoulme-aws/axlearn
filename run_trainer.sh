#!/usr/bin/env bash

# Editable paths

# CONDA
# CONDA_HOME="/shared/thangakr/conda"
# CONDA_ENV_NAME="tot"
# Source conda environment
# source ${CONDA_HOME}/bin/activate ${CONDA_ENV_NAME}

# VENV
PY_VENV_PATH="/home/apoorvgu/py310/bin/activate"
source ${PY_VENV_PATH}

NEURON_DUMP_PATH=${PWD}/temp3/
HLO_DUMP_PATH=./temp3/

echo "==============================================="
echo "Dependency versions"
echo "==============================================="
apt list | grep neuron | grep installed
pip freeze | grep neuron
echo "==============================================="

# Install runtime and collectives library. This is only needed in internal dev cluster
# Remove this before release
# source ./bigcluster_setup.sh
export XLA_FLAGS="--xla_dump_hlo_snapshots --xla_dump_to=/home/apoorvgu/axlearn/dump/ --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_to=./${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"
# export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_to=./${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"

# Neuron compiler flags
vnc_size=2
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2 --internal-num-neuroncores-per-sengine=${vnc_size} --internal-hlo2tensorizer-options='--verify-hlo --fuse-dot-logistic=false '"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"
export NEURON_RT_VIRTUAL_CORE_SIZE=${vnc_size}
export NEURON_RT_NUM_CORES=64
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_RT_DBG_DISABLE_PE_HW_DECODE=1

echo "vnc size is $vnc_size"
# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1

# Neuron env vars for distributed training based on SLURM
nodes=$(hostname)
# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
num_nodes=$(echo "$nodes" | wc -l)
process_idx=1
# process_idx=$(echo "$nodes" | grep -n "$SLURMD_NODENAME" | cut -d: -f1)
# devices_per_node=$NEURON_RT_NUM_CORES
# MASTER_ADDR=$(echo "$nodes" | head -n 1)
# MASTER_PORT=41000
# JAX_COORDINATOR_PORT=41001
# export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
# export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $NEURON_RT_NUM_CORES) | sed 's/,$//')
# export NEURON_PJRT_PROCESS_INDEX=$((process_idx - 1))
# export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
# export FI_LOG_LEVEL="warn"
# export FI_EFA_USE_DEVICE_RDMA="1"
# export FI_PROVIDER="efa"
# export FI_EFA_FORK_SAFE=1

OUTPUT_DIR="/home/apoorvgu/axlearn/out/"
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# Run the training script

python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v1 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=neuron-trn1.32xlarge-64
    # --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
    # --process_id=$NEURON_PJRT_PROCESS_INDEX
