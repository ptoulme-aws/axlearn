#! /bin/bash
set -e

sudo dpkg -i /fsx/mengchiy/TTOT/aws-neuronx-dkms_2.x.4095.0_amd64.deb
sudo dpkg -i /fsx/mengchiy/tot_binaries/aws-neuronx-collectives-2.x.21370.0-8cbb4877b.deb
sudo dpkg -i /fsx/mengchiy/tot_binaries/aws-neuronx-runtime-lib-2.x.19993.0-1bf746e12.deb


sudo apt-get -f install -y
sudo apt-get install -y google-perftools
# sudo apt -y --fix-broken install

# echo "after fix broken"
# sudo apt-get install -y google-perftools
# echo "after install"

# Neuron env vars for distributed training based on SLURM
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
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1

cd /fsx/mengchiy/axlearn_tests/axltest
pip list

# Editable paths
ARTIFACTS_PATH="../../artifacts8Bv2"
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}"
mkdir -p "$TEST_ARTIFACTS_PATH"

NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump

pwd #FIXME:
export PYTHONPATH="${PYTHONPATH}:${PWD}"
cd ..
export PYTHONPATH="${PYTHONPATH}:${PWD}"
cd axltest

python -c "import sys; print(sys.path)"

export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"


export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096 && export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2

# export NEURON_NUM_NODES
export NEURON_GRAD_ACC_COUNT=1

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1

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
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2" # --distribution-strategy=llm-training"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"

# fsdp_config=("fuji-70B-v2" "fuji-p70Bfsdp16tp4-v2" "fuji-p70Bfsdp4tp16-v2" "fuji-7Bfsdp16tp4-v2" "fuji-7Bfsdp64-v2" "fuji-70Bfsdp128tp4-v2" "fuji-7Bfsdp128tp4-v2", "fuji-p70Bfsdp32tp4-v2", "fuji-f7Bfsdp32tp4mb4-v2", "fuji-f70Bfsdp128tp4-v2", "fuji-f7Bfsdp128tp4mb4-v2", "fuji-p70Bfsdp8tp16-v2")

# if [[ " ${fsdp_config[@]} " =~ " $1 " ]]; then
#   export NEURON_FSDP=1
# fi
export NEURON_FSDP=1
export LNC=2
export ENABLE_NEW_UNSHARDED_ATTN_KERNEL=1

LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)

if [ -n "$LIBTCMALLOC" ]; then
    # Create a symbolic link to the found libtcmalloc version
    sudo ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
    echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"

    # Export LD_PRELOAD
    export LD_PRELOAD=/usr/lib/libtcmalloc.so
    echo "LD_PRELOAD set to: $LD_PRELOAD"
else
    echo "Error: libtcmalloc.so not found"
    exit 1
fi


MODULE="text.gpt.c4_trainer"
CONFIG=$1 # which config to use is passed into script as argument
BACKEND="neuron"
MESH="neuron-trn2n.48xlarge-64"

# trainer related directory #FIXME:
OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"

echo "Listing apt dependencies"
apt list --installed | grep neuron
echo "Listing pip dependencies"
pip list | grep neuron
echo "Done listing dependencies"

which python

echo "Listing hostname"
hostname

echo "printing env"
printenv | grep "NEURON"
printenv | grep "LD"

python -m axlearn.common.launch_trainer_main \
    --module=${MODULE} \
    --config=${CONFIG} \
    --trainer_dir=${OUTPUT_DIR} \
    --data_dir=${DATA_DIR} \
    --jax_backend=${BACKEND} \
    --mesh_selector=${MESH} \
    --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
    --process_id=$NEURON_PJRT_PROCESS_INDEX


#FIXME:
summary_folder="${OUTPUT_DIR}/summaries/train_train"

# Check if the folder exists and has exactly one file
if [ -d "$summary_folder" ]; then
    file=$(ls -1 "$summary_folder")
    if [ $(echo "$file" | wc -l) -eq 1 ]; then
        filepath="$summary_folder/$file"
        echo "file path: $filepath"
    fi
else
    echo "The summary folder ${summary_folder} does not exist."
    exit 1
fi

python check_step_time.py $filepath
