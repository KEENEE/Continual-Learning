# !/bin/bash

# Task Selection
TASK="mbpp2" # Available options: mbpp2, gsm8k, ai2_arc, cls

# Training Setting
NUM_ITERS=200

# This script needs 2 gpus
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \
    base_model@_global_=gemma4e4b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS
