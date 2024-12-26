#!/usr/bin/env bash

# Command line
# run_trainer_gpu.sh env_name testname test_setup
CONDA_ENV_NAME=$1
TESTNAME=$2
TEST_SETUP=$3

#RESUME_JOB_ID for checkpoint restore
if [[ -z "${RESUME_JOB_ID}" ]]; then
  JOB_ID=${SLURM_JOB_ID} 
else
  JOB_ID="${RESUME_JOB_ID}"
fi

#export NCCL_DEBUG="info" #For debugging NCCL and EFA 

# # Editable paths

# Source CONDA
CONDA_HOME="/home/${USER}/miniconda3/" #TODO : Make this $HOME once everyone has setup miniconda
source ${CONDA_HOME}/bin/activate ${CONDA_ENV_NAME}

which python
# # VENV
# PY_VENV_PATH="/shared/apoorvgu/jax-21/bin/activate"
# source ${PY_VENV_PATH}

# NEURON_DUMP_PATH=${PWD}/neuron_dump
# HLO_DUMP_PATH=${PWD}/hlo_dump

# # Install runtime and collectives library. This is only needed in internal dev cluster
# # Remove this before release
# source ./bigcluster_setup.sh

# # Neuron compiler flags
# export NEURON_CC_FLAGS="--framework=XLA"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --distribution-strategy=llm-training"
# # export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo --num-concat-graphs=8'" # Set indside fuji.py with gradient_accumulation size
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"

# # Neuron PJRT flags
# export NEURON_WHILE_LOOP_UNROLL=1
# export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1

# # Neuron runtime flags
# export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1

# Neuron env vars for distributed training based on SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
num_nodes=${SLURM_NNODES}
# devices_per_node=32
MASTER_ADDR=$(echo "$nodes" | head -n 1)
MASTER_PORT=41008
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
# export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=${SLURM_NODEID}
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_LOG_LEVEL="warn"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export LD_LIBRARY_PATH="$HOME/EFA/aws-ofi-nccl/lib:LD_LIBRARY_PATH"

ARTIFACTS_PATH="/fsx/${USER}/fs/runs/artifacts"
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/$CONDA_ENV_NAME/t_${TESTNAME}/${JOB_ID}"
mkdir -p "$TEST_ARTIFACTS_PATH"

echo "NEURON RUN DIRECTORY ${TEST_ARTIFACTS_PATH} ${SLURM_JOB_ID} ${RESUME_JOB_ID}"

NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# export JAX_PLATFORMS=cpu

#Perf Tuning Guideline here : https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/PGLE.md
#export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*' --xla_dump_hlo_as_proto"
:'
if [[ $GPU_RUN_TYPE == "profile" ]]; then
   echo "Executing GPU Profile Run"
   XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_disable_async_collectives=allreduce,allgather,reducescatter,collectivebroadcast,alltoall,collectivepermute"
elif [[ $GPU_RUN_TYPE == "perf" ]]; then
   echo "Executing GPU Performance Run"
   if [[ -z "${GPU_PROFILE}" ]]; then
       echo "ERROR : Can not run GPU Performance Run without a profile"
       exit 1
   fi
   XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
     --xla_gpu_enable_triton_gemm=false
     --xla_gpu_graph_level=0
     --xla_gpu_all_reduce_combine_threshold_bytes=1073741824
     --xla_gpu_all_gather_combine_threshold_bytes=1073741824
     --xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824
     --xla_gpu_enable_pipelined_all_gather=true
     --xla_gpu_enable_pipelined_reduce_scatter=true
     --xla_gpu_enable_pipelined_all_reduce=true
     --xla_gpu_enable_while_loop_double_buffering=true
     --xla_gpu_enable_all_gather_combine_by_dim=false
     --xla_gpu_enable_reduce_scatter_combine_by_dim=false
     --xla_disable_hlo_passes=rematerialization
     --xla_gpu_pgle_profile_file_or_directory_path=${GPU_PROFILE}
     ${XLA_FLAGS}" 
else
   #perf run with no profile
   echo "Executing GPU Run with no profile"
   XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_multi_streamed_windowed_einsum=true --xla_gpu_enable_custom_fusions=true --xla_gpu_enable_address_computation_fusion=true ${XLA_FLAGS}"
fi
'

###export NCCL_DEBUG=INFO
###export NCCL_DEBUG_SUBSYS=COLL

#HAH quick fix
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*' --xla_dump_hlo_as_proto --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_multi_streamed_windowed_einsum=true --xla_gpu_enable_custom_fusions=true" # --xla_gpu_enable_address_computation_fusion=true"

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p "$OUTPUT_DIR"
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# Run the training script

echo "distributed_coordinator " $NEURON_RT_ROOT_COMM_ID
echo "num_processes " $num_nodes
echo "process_id " $NEURON_PJRT_PROCESS_INDEX
hostname

source ${TEST_SETUP}
if [ "${N_EXPECTED_NODES}" -ne "${num_nodes}" ]; then
    echo "ERROR : ${TEST_SETUP} for ${N_EXPECTED_NODES} was launched with ${num_nodes}"
    exit 1
fi
MESH_SELECTOR="gpu-${num_nodes}node-baseline"

# python test.py
printenv  #Final env just before launch
python -u -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=$MODEL_ARCH \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=gpu --distributed_coordinator=$NEURON_RT_ROOT_COMM_ID \
    --num_processes=$num_nodes \
    --process_id=$NEURON_PJRT_PROCESS_INDEX \
    --mesh_selector=$MESH_SELECTOR

