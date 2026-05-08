# !/bin/bash

# Prerequisites:
#   - Ensure reference_params_results path is set in cfgs/base_model/*.yaml

# Task Selection
TASK="few_shot_math" # Available options: few_shot_arc_challenge, few_shot_humaneval

# Evaluation Setting
PER_LAYER=true
NORM_COEFFS=false

# Start evaluation!
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \
    base_model@_global_=llama3i8b \
    optimization@_global_=cem \
    policy@_global_=wcomb \
    use_loglikelihood_for_ties=true \
    per_layer=$PER_LAYER \
    task@_global_=$TASK \
    norm_coeffs=$NORM_COEFFS \
    wandb_log=true \
    num_iters=50 \
    "$@"