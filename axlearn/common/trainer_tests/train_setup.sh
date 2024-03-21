#!/usr/bin/env bash
set -o pipefail
set -e

ulimit -n 65535
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
sudo sysctl -w net.ipv4.ip_local_reserved_ports=41000
if which lctl >/dev/null 2>&1; then
    sudo lctl set_param 'osc.*.max_dirty_mb=64' # Cap max space each connection to FSx reserves so we avoid OODs
fi
IPS=""
for h in $(scontrol show hostname); do
    IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
done
HOSTS=(${IPS//\ / })
NODEID=$SLURM_NODEID
NTASKS=$SLURM_NTASKS
export PROCESSES_PER_NODE=1
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"

export NEURON_RT_EXEC_TIMEOUT=100
export DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

#export XLA_USE_SPMD=1
#export TF_CPP_MIN_LOG_LEVEL=0 # Enable SPMD verbose logging - 0 means most verbose
#export TF_CPP_MAX_VLOG_LEVEL=2 # Needs above flag for logging but goes in reverse. 0 means no log
#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_MAX_VLOG_LEVEL=5

export PJRT_DEVICE="NEURON"
export NEURON_RT_NUM_CORES=32
export NEURON_PJRT_PROCESS_INDEX=$NODEID
export RANK=$NODEID
export PJRT_LOCAL_PROCESS_COUNT=1
export WORLD_SIZE=$((NTASKS * 32))
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '32,'%.0s $(seq 1 $NTASKS) | sed 's/,$//')
export NEURON_COMPILE_CACHE_URL="/shared/ptoulme/jax/jax_cache"
# For FP32
#export NEURON_CC_FLAGS="--auto-cast=none --retry_failed_compilation"
export NEURON_INTERNAL_USE_VANILLA_TORCH_XLA=1
export NEURON_USE_VANILLA_TORCH_XLA=1
export NEURON_TRANSFER_WITH_STATIC_RING_OPS=""
export NEURON_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING=0
export XLA_FLAGS="--xla_force_host_platform_device_count=32 --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_to=./jax_dump_new --xla_dump_hlo_pass_re='.*'"

#Snapshotting
#export XLA_FLAGS=" --xla_dump_hlo_snapshots --xla_dump_to=/shared/ptoulme/GSPMD/NeuronGSPMDTests/src/NeuronGSPMDTests/transformers/snapshots"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

#Runtime debug
#export NEURON_RT_LOG_LEVEL_NRT = 'DEBUG'
# BF16
export XLA_USE_BF16=1
#export NEURON_CC_FLAGS="--dump=./compiler_dump --framework=XLA --model-type=transformer --distribution-strategy=llm-training -O1 --no-internal-hlo-remat"
export NEURON_CC_FLAGS="--dump=./compiler_dump_c4 --framework=XLA --model-type transformer --no-internal-hlo-remat --enable-mixed-precision-accumulation -O1"

export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=5

export JAX_TRACEBACK_FILTERING=off # this enables verbose frame logging in jax

# llm training