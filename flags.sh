NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
# --xla_dump_hlo_as_proto
# --xla_dump_hlo_snapshots
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"
export TF_CPP_MIN_LOG_LEVEL=0

# Check if VNC variable is set and valid
if [[ -z "${VNC}" || ! "${VNC}" =~ ^[12]$ ]]; then
    echo "Error: VNC variable must be set to either 1 or 2" >&2
    exit 1
fi
vnc_size=${VNC}

# Neuron compiler flags
export NEURON_RT_VIRTUAL_CORE_SIZE=${vnc_size}
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
#--internal-compiler-debug-mode=all
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=${vnc_size}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo --fuse-dot-logistic=false'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --distribution-strategy=llm-training"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-compiler-debug-mode=penguin"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"
#export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --disable-internal-dge"

# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
export TRN2=1
# export NEURON_TRANSFORMER_SHARDING=1

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_EXEC_TIMEOUT=300

# Neuron env vars for distributed training based on SLURM
nodes=$SLURM_NODES

# Print nodes for diagnostic purposes
echo "Nodes: $nodes"
num_nodes=$(echo "$nodes" | wc -w)
echo "Number of nodes: $num_nodes"
echo "SLURMD_NODENAME: $SLURMD_NODENAME"

# Compute process_idx as the index of SLURMD_NODENAME in SLURM_NODES
process_idx=$(echo "$nodes" | tr ' ' '\n' | awk -v node="$SLURMD_NODENAME" '{if ($0 == node) print NR}')

devices_per_node=$((128 / vnc_size))
MASTER_ADDR=$(echo "$nodes" | cut -d' ' -f1)
MASTER_PORT=41000
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf "%s," $(for i in $(seq 1 $num_nodes); do echo $devices_per_node; done) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$((process_idx - 1))
echo "NEURON_PJRT_PROCESS_INDEX: $NEURON_PJRT_PROCESS_INDEX"
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_LOG_LEVEL="warn"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1

print all env vars
set
