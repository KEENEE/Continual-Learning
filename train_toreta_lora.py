"""Standalone LoRA training with REINFORCE policy gradient.

Uses HuggingFace PEFT for LoRA and vLLM for fast generation.
Gradients flow through a single autograd graph — no custom backward().

Usage:
    python train_toreta_lora.py --task_name mbpp2 --model_id google/gemma-4-E4B-it
    # or via the launcher script:
    bash scripts/train_toreta_lora.sh mbpp2 200
"""

import argparse
import gc
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import Metrics, get_mean_std_max_min_dict
from tasks import AI2ArcTask, Gsm8kTask, Mbpp2Task
from utils import eval_model, load_hf_params_to_vllm

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "mbpp2": Mbpp2Task,
    "gsm8k": Gsm8kTask,
    "ai2_arc": AI2ArcTask,
}


def load_task(task_name):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_name]()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_hf_model(model_id, gpu):
    """Load HF model with Gemma4-specific handling."""
    is_gemma4 = "gemma-4" in model_id.lower()
    if is_gemma4:
        from transformers import Gemma4ForConditionalGeneration
        model_cls = Gemma4ForConditionalGeneration
    else:
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_id, device_map=gpu, torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if is_gemma4:
        inner = model.model
        for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            if getattr(inner, attr, None) is not None:
                setattr(inner, attr, None)
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def apply_lora(model, args):
    """Wrap model with PEFT LoRA adapters."""
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


# ---------------------------------------------------------------------------
# vLLM weight sync
# ---------------------------------------------------------------------------
def merge_and_push_to_vllm(peft_model, vllm_model):
    """Merge LoRA into base weights, push to vLLM, then unmerge."""
    peft_model.merge_adapter()
    # Get the underlying model's state_dict (without PEFT wrapper keys)
    state_dict = peft_model.base_model.model.state_dict()
    load_hf_params_to_vllm(state_dict, vllm_model.llm)
    peft_model.unmerge_adapter()


# ---------------------------------------------------------------------------
# REINFORCE training step
# ---------------------------------------------------------------------------
def train_step(
    peft_model,
    tokenizer,
    vllm_model,
    task_loader,
    train_data,
    train_eval,
    batch_ix,
    optimizer,
    args,
    metrics,
    gpu,
):
    model_id = args.model_id

    # 1. Build prompts
    prompts = [
        task_loader.get_prompt(tokenizer, train_data, i, model_id=model_id)
        for i in batch_ix
    ]
    batch_size = len(prompts)

    # 2. Merge LoRA and push to vLLM for generation
    print("Merging LoRA and loading weights to vLLM...")
    merge_and_push_to_vllm(peft_model, vllm_model)

    # 3. Generate completions via vLLM
    print("Generating completions with vLLM...")
    res = eval_model(vllm_model, train_eval, batch_ix)

    # 4. Compute rewards
    rewards = np.array(task_loader.get_rewards(res=res))
    if args.rw_norm:
        mean_rw = np.mean(rewards)
        std_rw = np.clip(np.std(rewards), 1e-7, None)
        rewards = (rewards - mean_rw) / std_rw
    if args.rw_clip is not None and args.rw_clip > 0:
        rewards = np.clip(rewards, -args.rw_clip, args.rw_clip)
    metrics.update(**get_mean_std_max_min_dict(rewards, "rewards"))

    # 5. Optional: reference log-probs for KL penalty
    ref_log_probs_list = None
    if args.kl_ref_coeff > 0:
        ref_log_probs_list = []
        with torch.no_grad():
            peft_model.disable_adapter_layers()
            for j, prompt in enumerate(prompts):
                output_text = res.sample_details[j]["output"]
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(gpu)
                prompt_length = input_ids.shape[-1]
                full_ids = tokenizer(
                    prompt + output_text, return_tensors="pt"
                ).input_ids.to(gpu)
                outputs = peft_model(full_ids)
                logits = outputs.logits[:, prompt_length - 1 : -1]
                ref_lp = F.log_softmax(logits, dim=-1)
                ref_log_probs_list.append(ref_lp.detach().cpu())
            peft_model.enable_adapter_layers()

    # 6. Compute REINFORCE loss with direct gradient flow
    print("Computing policy gradient...")
    for j, prompt in enumerate(prompts):
        output_text = res.sample_details[j]["output"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(gpu)
        prompt_length = input_ids.shape[-1]
        full_ids = tokenizer(
            prompt + output_text, return_tensors="pt"
        ).input_ids.to(gpu)
        generated_ids = full_ids[:, prompt_length:]

        outputs = peft_model(full_ids)
        logits = outputs.logits[:, prompt_length - 1 : -1]
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(
            2, generated_ids.unsqueeze(-1)
        ).squeeze(-1)
        log_likelihood = selected_log_probs.sum(dim=-1)

        pg_loss = -log_likelihood * rewards[j]
        loss = pg_loss

        if args.kl_ref_coeff > 0 and ref_log_probs_list is not None:
            ref_lp = ref_log_probs_list[j].to(gpu)
            kl_div = F.kl_div(
                log_probs, ref_lp, log_target=True, reduction="sum"
            )
            loss = loss + args.kl_ref_coeff * kl_div

        scaled_loss = loss / batch_size
        scaled_loss.backward()

        metrics.update(pg=pg_loss.item(), loss=loss.item())

    # 7. Clip and step
    torch.nn.utils.clip_grad_norm_(
        [p for p in peft_model.parameters() if p.requires_grad],
        args.max_grad_norm,
    )
    optimizer.step()
    optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    task_loader = load_task(args.task_name)
    gpu = torch.device(f"cuda:{args.hf_gpu}")

    # Load models
    print(f"Loading HF model: {args.model_id}")
    model, tokenizer = load_hf_model(args.model_id, gpu)
    peft_model = apply_lora(model, args)

    print("Loading vLLM model...")
    vllm_model = task_loader.get_vllm_model(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Data
    train_data, train_ix, valid_ix = task_loader.get_train_data()
    train_eval, *test_evals = task_loader.get_evaluator()
    test_eval = test_evals[0]
    transfer_eval = test_evals[1] if task_loader.has_transfer_split else None

    # Optimizer
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # Logging
    metrics = Metrics()
    exp_name = args.exp_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        f"{args.out_dir}/{args.task_name}/toreta_lora_r{args.lora_rank}"
        f"_a{args.lora_alpha}/{exp_name}"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    wandb = None
    if args.wandb_log:
        import wandb as _wandb
        _wandb.init(
            project=args.wandb_project,
            name=f"{args.task_name}_lora_r{args.lora_rank}_{exp_name}"[:127],
            config=vars(args),
        )
        wandb = _wandb

    # Baseline eval
    print("=== Baseline eval (base model, no LoRA effect) ===")
    baseline_metrics = {}
    b_test = eval_model(vllm_model, test_eval)
    baseline_metrics["baseline_test_acc"] = b_test.aggregate_metrics[
        task_loader.target_metric_test
    ]
    if task_loader.has_training_split:
        b_train = eval_model(vllm_model, train_eval, train_ix)
        b_valid = eval_model(vllm_model, train_eval, valid_ix)
        baseline_metrics["baseline_train_acc"] = b_train.aggregate_metrics[
            task_loader.target_metric_train
        ]
        baseline_metrics["baseline_valid_acc"] = b_valid.aggregate_metrics[
            task_loader.target_metric_valid
        ]
    if transfer_eval is not None:
        b_transfer = eval_model(vllm_model, transfer_eval)
        baseline_metrics["baseline_transfer_acc"] = b_transfer.aggregate_metrics[
            task_loader.target_metric_transfer
        ]
    print(f"Baseline: {baseline_metrics}")
    with open(f"{log_dir}/baseline_eval.json", "w") as f:
        json.dump(baseline_metrics, f, indent=4)
    if wandb:
        wandb.log(baseline_metrics)

    # Training loop
    np_random = np.random.RandomState(args.seed)
    clipped_batch_size = min(args.batch_size, len(list(train_ix)))
    best_val_acc = 0.0
    test_at_best = 0.0
    transfer_at_best = 0.0

    for i in range(args.num_iters):
        batch_ix = np_random.choice(train_ix, size=clipped_batch_size, replace=False)

        train_step(
            peft_model, tokenizer, vllm_model, task_loader,
            train_data, train_eval, batch_ix, optimizer,
            args, metrics, gpu,
        )

        gc.collect()
        torch.cuda.empty_cache()

        # Log gradient stats
        with torch.no_grad():
            grad_mags = []
            param_mags = []
            for p in trainable_params:
                if p.grad is not None:
                    grad_mags.append(torch.linalg.vector_norm(p.grad).item())
                param_mags.append(torch.linalg.vector_norm(p).item())
            if grad_mags:
                metrics.update(
                    **get_mean_std_max_min_dict(grad_mags, "grad_mags")
                )
            metrics.update(
                **get_mean_std_max_min_dict(param_mags, "param_mags")
            )

        metrics_dict = metrics.get()
        grad_mag = metrics_dict.get("grad_mags/mean", None)
        pg = metrics_dict.get("pg", None)
        print(f"Iter {i}: grad_mag={grad_mag}, PG={pg}")

        # Periodic evaluation
        if i % args.test_interval == 0:
            merge_and_push_to_vllm(peft_model, vllm_model)

            train_res = eval_model(vllm_model, train_eval, train_ix)
            valid_res = eval_model(vllm_model, train_eval, valid_ix)
            test_res = eval_model(vllm_model, test_eval)

            valid_acc = valid_res.aggregate_metrics[task_loader.target_metric_valid]
            test_acc = test_res.aggregate_metrics[task_loader.target_metric_test]
            train_acc = train_res.aggregate_metrics[task_loader.target_metric_train]

            data_dict = {
                "iter": i,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "test_acc": test_acc,
                "best_val_acc": best_val_acc,
                "test_at_best_val": test_at_best,
                **metrics_dict,
            }

            if transfer_eval is not None:
                transfer_res = eval_model(vllm_model, transfer_eval)
                data_dict["transfer_acc"] = transfer_res.aggregate_metrics[
                    task_loader.target_metric_transfer
                ]

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                test_at_best = test_acc
                if transfer_eval is not None:
                    transfer_at_best = data_dict.get("transfer_acc", 0.0)
                data_dict["best_val_acc"] = best_val_acc
                data_dict["test_at_best_val"] = test_at_best
                print("best_val_acc updated — saving adapter")
                if args.save_adapter:
                    peft_model.save_pretrained(f"{log_dir}/best_adapter")
                    torch.save(optimizer.state_dict(), f"{log_dir}/best_optimizer.pt")

            # Always save latest
            if args.save_adapter:
                peft_model.save_pretrained(f"{log_dir}/latest_adapter")

            print(
                f"Iter {i} eval | "
                f"train={train_acc:.4f} | valid={valid_acc:.4f} | "
                f"test={test_acc:.4f} | "
                f"best_val={best_val_acc:.4f} (test@best={test_at_best:.4f})"
            )
            if wandb:
                wandb.log(data_dict)
            with open(f"{log_dir}/training_log.json", "a") as f:
                f.write(json.dumps(data_dict, indent=4) + "\n")

            metrics.reset()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LoRA + REINFORCE training")

    # Model
    p.add_argument("--model_id", default="google/gemma-4-E4B-it")
    p.add_argument("--hf_gpu", type=int, default=1)
    p.add_argument("--tensor_parallel_size", type=int, default=1)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--num_iters", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Reward processing
    p.add_argument("--rw_norm", action="store_true")
    p.add_argument("--rw_clip", type=float, default=None)

    # KL regularization
    p.add_argument("--kl_ref_coeff", type=float, default=0.0)

    # Task
    p.add_argument("--task_name", default="mbpp2",
                    choices=list(TASK_REGISTRY.keys()))

    # Evaluation
    p.add_argument("--test_interval", type=int, default=5)

    # Logging
    p.add_argument("--out_dir", default="results")
    p.add_argument("--exp_name", default=None)
    p.add_argument("--wandb_log", action="store_true")
    p.add_argument("--wandb_project", default="toreta-lora")

    # Checkpointing
    p.add_argument("--save_adapter", action="store_true", default=True)
    p.add_argument("--no_save_adapter", dest="save_adapter", action="store_false")

    return p.parse_args()


if __name__ == "__main__":
    main()
