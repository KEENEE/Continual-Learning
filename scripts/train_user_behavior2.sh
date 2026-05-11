# !/bin/bash

# Available base models: llama3i8b, llama31i8b, gemma4e4bi
# Available optimizers : reinforce (RL), sft (supervised), rsm, cem

BASE_MODEL="gemma4e4bi"
OPTIM="sft"
NUM_ITERS=200
BATCH_SIZE=8
LEARNING_RATE=5e-4

export WANDB_MODE=disabled
export USE_FASTSAFETENSOR=true
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

CUDA_VISIBLE_DEVICES=2,3 python svd_reinforce_hydra.py \
    base_model@_global_=$BASE_MODEL \
    task@_global_=user_behavior \
    optimization@_global_=$OPTIM \
    num_iters=$NUM_ITERS \
    batch_size=$BATCH_SIZE \
    lr=$LEARNING_RATE \
    mode@_global_=training \
    test_only=false
