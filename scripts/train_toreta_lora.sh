#!/bin/bash

# Usage: bash scripts/train_toreta_lora.sh [TASK] [NUM_ITERS] [extra args...]
# Examples:
#   bash scripts/train_toreta_lora.sh mbpp2 200
#   bash scripts/train_toreta_lora.sh gsm8k 200 --rw_norm
#   bash scripts/train_toreta_lora.sh ai2_arc 200 --lr 1e-4

TASK="${1:-mbpp2}"
NUM_ITERS="${2:-200}"

HF_HOME=/group-volume/mjkwon/cache/huggingface \
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
USE_FASTSAFETENSOR=true \
python train_toreta_lora.py \
    --model_id google/gemma-4-E4B-it \
    --task_name "$TASK" \
    --num_iters "$NUM_ITERS" \
    --lr 5e-5 \
    --max_grad_norm 1.0 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --test_interval 5 \
    "${@:3}"
