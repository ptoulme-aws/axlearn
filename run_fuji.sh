OUTPUT_DIR="/home/ubuntu/"
DATA_DIR="FAKE"
export JAX_PLATFORMS=cpu
python -u /home/ubuntu/axlearn/axlearn/common/launch_trainer_main.py \
    --module=text.gpt.c4_trainer --config=fuji-test \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR --jax_backend=cpu


