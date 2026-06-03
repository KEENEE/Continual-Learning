# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase extending **Transformer² (Self-adaptive LLMs)** — see `README.md`
and the paper (arXiv:2501.06252). The core idea: instead of LoRA-style additive
adapters, **SVD-decompose every 2-D weight matrix** of a base LLM into `U diag(S) Vᵀ`,
then learn a small per-singular-value **mask** that rescales the singular values. These
masks are the "expert vectors." At inference a dispatcher mixes experts to adapt to a
task without full fine-tuning.

The `Continual-Learning` fork adds a **`user_behavior`** task: predicting a phone user's
next action from their usage history, with a data-generation pipeline (`1_filter.py` …
`4_extract.py`) and supervised (SFT) training on top of the original RL machinery.

## Setup & common commands

```bash
# Environment (Python 3.11)
conda create -n t2 python=3.11 -y && conda activate t2
pip install --upgrade pip
pip install -r requirements.txt

# REQUIRED: install the vendored evaluator package (editable)
cd evaluation/fishfarm && pip install -e . && cd -
```

Everything trains/evals through one Hydra entry point, `svd_reinforce_hydra.py`.
Wrapper scripts in `scripts/` set sensible flags:

```bash
bash scripts/train_task_expert.sh        # RL-train an expert (mbpp2/gsm8k/ai2_arc/cls) on Llama-3-8B
bash scripts/train_task_expert_gemma.sh  # same, Gemma-4 (needs VLLM_ALLOW_INSECURE_SERIALIZATION=1)
bash scripts/train_user_behavior.sh      # SFT-train the user_behavior task on Gemma-4
bash scripts/eval_prompt_based.sh        # 2-pass dispatch eval: classify prompt -> route to expert
bash scripts/eval_few_shot.sh            # few-shot eval mixing pre-trained experts (CEM + wcomb)
```

To run a single configuration directly, override Hydra groups/keys on the CLI:

```bash
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \
    base_model@_global_=llama3i8b task@_global_=gsm8k \
    optimization@_global_=reinforce mode@_global_=training \
    num_iters=200 wandb_log=false
```

- **First run per base model is slow**: it performs the SVD decomposition and writes
  `<model>_decomposed_params.pt` to the repo root, then `return`s without training.
  Re-run the same command to train. Force re-extraction with `extract_svd=true`.
- **Two GPUs are required.** The HF model (gradients) is hard-pinned to `cuda:1` in
  `svd_reinforce_hydra.py`; the vLLM generation engine takes `cuda:0`. Always pass
  `CUDA_VISIBLE_DEVICES=0,1` (or two visible devices).
- Disable Weights & Biases per-run with `wandb_log=false` or `WANDB_MODE=disabled`.
- There is **no test suite or linter** at the repo root. `evaluation/fishfarm` ships its
  own `tox.ini`/`pyproject.toml` (black, isort, flake8, mypy, pytest) for that subpackage only.

## Architecture — the big picture

### Train/eval loop (`svd_reinforce_hydra.py`)
The single `main()` orchestrates everything via `hydra.utils.instantiate` on five config
groups (see `cfgs/config.yaml`): `task_loader`, `base_model`, `shakeoff_policy`,
`optimization_algorithm`, and `mode`. Flow:
1. Load base model on `cuda:1`; load (or compute) its SVD decomposition.
2. Build a vLLM model for fast generation (via the **task**, not directly).
3. Optional baseline eval, then either a `test_only` eval branch or the training loop.
4. Training loop calls `optimization_algorithm.step_optimization(...)` then `.update(...)`
   per iteration; every `test_interval` it materializes current masks into weights, pushes
   them to vLLM, evaluates train/valid/test, and checkpoints on best valid accuracy.

### The mask ↔ weight bridge (`utils.py`) — central to understanding the code
- `compose_new_params` reconstructs a weight as
  `U @ diag(S · mask) @ Vᵀ`, **renormalized** by `S.sum() / (S·mask).sum()` so total
  spectral energy is preserved.
- `forward()` writes composed weights into the live HF model (no_grad).
- `backward()` is a **manual** reverse pass: gradients accumulate on the materialized
  HF weight tensors, then `compose_new_params(...).backward(weight.grad)` propagates them
  back into the learnable masks. This is why optimizers call `forward(...)` /
  `backward(...)` explicitly rather than relying on a normal autograd graph through vLLM.
- `load_hf_params_to_vllm` ships updated weights into the running vLLM engine. It handles
  vLLM's packed layouts (`q/k/v_proj → qkv_proj`, `gate/up_proj → gate_up_proj`) and, on
  vLLM **v1**, sends **one layer per `apply_model` IPC call** to stay under the 2³²-byte
  msgspec cap.
- Convention used everywhere: only params with `"norm" not in k and "embed" not in k and
  v.ndim >= 2` are decomposed/masked. Changing this filter must be done consistently in
  `policy/base.py`, `utils.py`, and `svd_reinforce_hydra.py`.

### Optimizers (`optim_modules.py`)
All subclass `OptimizationAlgorithm` with `step_optimization` + `update`:
- **`Reinforce`** — RL policy gradient. Rewards from `task_loader.get_rewards`; optional
  reward norm/clip and KL-to-reference penalty. Uses real gradients via `backward()`.
- **`SupervisedSFT`** — minimizes `-log p(gold | prompt)`; gold from
  `task_loader.get_gold_target`. Same gradient path as Reinforce. Respects a per-task
  `train_max_len` cap to avoid OOM on long prompts / large vocabs.
- **`RandomShooting` (rsm)** and **`CEM`** — gradient-free; sample a population of masks,
  evaluate each with vLLM, keep best / fit elites. No `backward()` needed.

### Policies (`policy/`)
- `Policy` (base): one learnable vector per matrix; mask = `sigmoid(p) * max_mult`. This is
  the **first-pass / single-expert** training object.
- `WeightedCombination` (`wcomb`): loads several pre-trained expert checkpoints and learns
  mixing coefficients over them (optionally per-layer / normalized). This is the
  **second-pass dispatch** used in few-shot eval.

### Tasks (`tasks/`)
Each `Task` provides train data, an evaluator, prompt formatting, rewards, and a configured
vLLM model. Built-in: `gsm8k`, `math`, `mbpp2`, `ai2_arc`, `cls`, plus `FewShotTask`
(wraps any task for k-shot eval) and `user_behavior`. Evaluators come from the vendored
`fishfarm` library, except `user_behavior` which uses its own
`UserBehaviorEvaluator` (sentence-transformer cosine similarity: `sem_acc` = fraction with
cosine ≥ threshold; reward = `2·sim − 1`).

### Base models (`base_model/`)
Thin wrappers mapping a HF model id to its decomposed-params filename: `llama3i8b`,
`llama31i8b`, `mistral03i7b`, `gemma4e4bi`. **Gemma-4 is special-cased throughout**
(`svd_reinforce_hydra.py`, `utils.py`, `tasks/user_behavior.py`): the checkpoint is a
multimodal `Gemma4ForConditionalGeneration` whose text weights live under
`model.language_model.*`. The code loads the multimodal class, deletes the vision/audio
towers, strips the prefix when syncing to vLLM, and forces vLLM to the text-only
`Gemma4ForCausalLM`. Don't "simplify" these branches without re-checking weight-key matching.

## user_behavior data pipeline

The numbered scripts at the repo root are a standalone preprocessing chain that produces
`data/user_behavior_samples.json` (consumed by `tasks/user_behavior.py`):

`1_filter.py` (drop apps/notifications by keyword) → `2_compress.py` (shrink the lifelog
JSON for prompting) → `3_abstract.py` (one-line semantic event descriptions) →
`4_extract.py` (mine `(trigger, context, gold)` preference patterns → train/valid/test +
`continual` phase splits). `4_extract_planeat.py` is a variant. The algorithm and its
revisions are documented in `plan.md` (English) and `plan_kr.md` (Korean) — read `plan.md`
before modifying `4_extract.py`. `data_generation_old/` holds the frozen Phase-1 version.

**All user data is git-ignored** (`data/`, `user_usage_*`, `*_samples.json`, etc. — see
`.gitignore`). Treat raw usage logs as private; never commit them.

## Conventions & gotchas

- **Config-driven.** Prefer adding a YAML under `cfgs/<group>/` and a class wired via
  `_target_` over hardcoding. New tasks/policies/optimizers must be exported in the
  package `__init__.py` and given a matching `cfgs/` entry.
- Checkpoints: `policy_params.pt` (best valid) and `policy_params_latest.pt` are the
  policy `state_dict`; `learnable_params*.pt` (legacy, gated by `save_legacy_params`) are
  the raw mask tensors. Loading logic in `main()` branches on whether the path contains
  `"learnable_params"`.
- Generated artifacts and run outputs (`results/`, `results_eval/`, `wandb/`,
  `saved_models/`, `*_decomposed_params.pt`) are git-ignored; don't commit them.
- Gemma-4 / vLLM-v1 runs need `VLLM_ALLOW_INSECURE_SERIALIZATION=1` so `apply_model` can
  ship closures over IPC (see `scripts/train_*_gemma.sh` / `train_user_behavior.sh`).
