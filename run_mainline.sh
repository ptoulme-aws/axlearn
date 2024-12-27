#!/usr/bin/env bash
# Neuron env vars for distributed training based on SLURM
export DATA_SEED=42
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
num_nodes=$(echo "$nodes" | wc -l)
devices_per_node=64
MASTER_ADDR=$(echo "$nodes" | head -n 1)
MASTER_PORT=41000
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$SLURM_NODEID
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_LOG_LEVEL="warn"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1

#install neuron libraries
sudo dpkg -i /path_to/aws-neuronx-runtime-lib-2.x.19946.0-471ad7ed2.deb
sudo dpkg -i /path_to/aws-neuronx-collectives-2.x.21304.0-36af3830a.deb
sudo dpkg -i /path_to/axlearn/aws-neuronx-dkms_2.x.3951.0_amd64.deb
hostname
ARTIFACTS_PATH=<artifacts path>
TIMESTAMP=$(date +"%y%m%d%H%M%S")
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${TIMESTAMP}"
mkdir -p "$TEST_ARTIFACTS_PATH"
NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives --xla_dump_hlo_as_proto --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"
# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1
export NEURON_ALL_REDUCE_UPCASTER=1
# Neuron collectives flag
export FI_LOG_LEVEL="warn"
export OFI_NCCL_PROTOCOL=RDMA
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export OFI_NCCL_MR_CACHE_DISABLE=1
# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2" # --distribution-strategy=llm-training"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2 --internal-hlo2tensorizer-options='--verify-hlo'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_FSDP=1
export LNC=2
export NEURON_ALL_REDUCE_UPCASTER=1
# conda
<activate conda environment>

echo "Listing apt dependencies"
apt list --installed | grep neuron
echo "Listing pip dependencies"
pip list | grep neuron
echo "Done listing dependencies"
which python
OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
python -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-70B-v2 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=neuron-trn2.48xlarge-64 \
    --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
    --process_id=$NEURON_PJRT_PROCESS_INDEX