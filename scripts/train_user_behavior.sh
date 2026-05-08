# !/bin/bash

# Train the user-behavior task. Choose base model and optimizer below.
# Available base models: llama3i8b, llama31i8b, gemma4e4b
# Available optimizers : reinforce (RL), sft (supervised), rsm, cem
BASE_MODEL="llama3i8b"
OPTIM="sft"
NUM_ITERS=200
BATCH_SIZE=8

# This script needs 2 gpus
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \
    base_model@_global_=$BASE_MODEL \
    task@_global_=user_behavior \
    optimization@_global_=$OPTIM \
    mode@_global_=training \
    num_iters=$NUM_ITERS \
    batch_size=$BATCH_SIZE \
    test_only=false \
    "$@"
