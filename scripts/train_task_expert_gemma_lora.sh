# !/bin/bash

# Task Selection
TASK="mbpp2" # Available options: mbpp2, gsm8k, ai2_arc, cls

# Training Setting
NUM_ITERS=200

# This script needs 2 gpus.
# VLLM_ALLOW_INSECURE_SERIALIZATION=1 lets vLLM v1's IPC pickle the
# closures we use to push weights into the engine (apply_model passes
# a lambda + tensor dict; default msgpack can't encode either).
CUDA_VISIBLE_DEVICES=0,1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 python svd_reinforce_hydra.py \
    base_model@_global_=gemma4e4bi \
    policy@_global_=lora \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS \
    "$@"
