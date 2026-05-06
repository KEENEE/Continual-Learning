import gc
import json
import os
from datetime import datetime
from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from base_model import BaseModel
from logging_utils import Metrics, get_mean_std_max_min_dict
from optim_modules import OptimizationAlgorithm
from policy import Policy
from tasks import Task
from utils import (eval_model, eval_model_experts_prompt_based, forward,
                   load_hf_params_to_vllm)


def wandb_init(cfg, run_name: str, group_name: str, log_dir: str):
    import wandb

    config_dict = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=False,
    )
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    # wandb has a 128-size character limit on the group name
    wandb.init(
        project=cfg.wandb_project,
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )
    return wandb


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg):
    """Main function."""

    num_iters = cfg.num_iters
    test_interval = cfg.test_interval

    batch_size = cfg.batch_size
    seed = cfg.seed
    policy_name = cfg.policy_name
    test_only = cfg.test_only
    save_legacy_params = cfg.save_legacy_params
    exp_name = cfg.exp_name
    run_name = cfg.run_name

    task_name = cfg.task_name

    load_ckpt = cfg.load_ckpt
    use_lora = cfg.use_lora
    prompt_based_eval = cfg.prompt_based_eval
    experts_path_dict = cfg.experts_path_dict

    resuming_from_ckpt = False
    if load_ckpt is not None:
        if load_ckpt == "scratch" or load_ckpt == "base":
            resuming_from_ckpt = False
        else:
            resuming_from_ckpt = True

    # Create task
    task_loader: Task = hydra.utils.instantiate(cfg.task_loader)

    base_model: BaseModel = hydra.utils.instantiate(cfg.base_model)

    model_id = base_model.get_model_id()
    decomposed_param_file = base_model.get_param_file(param_folder_path="")

    extract_svd = cfg.extract_svd or (not os.path.exists(decomposed_param_file))

    has_training_split = task_loader.has_training_split
    has_transfer_split = task_loader.has_transfer_split

    if not has_training_split:
        assert test_only, "Cannot train on a task with no training split"

    if exp_name is None:
        exp_name = "temp"

    metrics_to_log = Metrics()

    # Create log dir.
    if run_name is None:
        now = datetime.now()
        run_name = now.strftime("%Y%m%d-%H%M%S")
    if test_only and (not resuming_from_ckpt):
        log_dir = f"{cfg.out_dir}/{task_name}/{cfg.base_model_name}_base"
        group_name = cfg.base_model_name
    else:
        log_dir = f"{cfg.out_dir}/{task_name}/{policy_name}/{exp_name}/{run_name}"
        group_name = cfg.wandb_group_name
    os.makedirs(log_dir, exist_ok=True)

    vllm_model = task_loader.get_vllm_model(
        model_id=model_id,
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
    )

    train_eval, *test_evals = task_loader.get_evaluator()
    if task_loader.has_transfer_split:
        test_eval, transfer_eval = test_evals
    else:
        test_eval = test_evals[0]

    train_data, train_ix, valid_ix = task_loader.get_train_data()
    gpu = torch.device("cuda:1")
    np_random = np.random.RandomState(seed)

    # The Gemma 4 checkpoint was saved from Gemma4ForConditionalGeneration:
    # the text submodule lives at model.language_model.*. Loading via the
    # text-only Gemma4ForCausalLM (which expects model.*) silently misses
    # the prefix, so transformers reports every weight UNEXPECTED/MISSING
    # and re-initializes the whole model to random — SVD then trips on NaN.
    # Load the multimodal class so the checkpoint actually populates, then
    # delete vision/audio submodules below to reclaim VRAM.
    is_gemma4 = "gemma-4" in model_id.lower()
    if is_gemma4:
        from transformers import Gemma4ForConditionalGeneration
        model_cls = Gemma4ForConditionalGeneration
    else:
        model_cls = AutoModelForCausalLM

    # GPU + float32 for initial SVD decomposition (cuBLAS-accelerated svd)
    if extract_svd:
        model = model_cls.from_pretrained(
            model_id, device_map=gpu, torch_dtype=torch.float32
        )
    else:
        # Load model and tokenizer.
        model = model_cls.from_pretrained(
            model_id, device_map=gpu, torch_dtype=torch.bfloat16
        )
        # Gradient checkpointing trades ~30% compute for ~50-60% activation
        # memory savings during backward — required for long-context training
        # (e.g., user_behavior 7K-token prompts) on a single 80GB GPU.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if is_gemma4:
        # Drop multimodal towers we never use so they don't end up in
        # state_dict (would feed vision/audio MLPs into the SVD loop) or
        # consume VRAM during training.
        inner = model.model
        for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            if getattr(inner, attr, None) is not None:
                setattr(inner, attr, None)
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_params = model.state_dict()

    original_model_params = {
        k: v.clone().detach().cpu()
        for k, v in base_params.items()
        if "mlp" in k and v.ndim >= 2
    }

    # Load decomposed parameters.
    if not os.path.exists(decomposed_param_file):
        print("Decomposed params not found. Decomposing on GPU...")
        decomposed_params = {}
        for k, v in base_params.items():
            # Gemma 4's Gemma4ClippableLinear registers 0-D buffers
            # (input_min/max, output_min/max) that pass the "norm not in k"
            # check but aren't 2-D matrices. Require >=2-D explicitly.
            if "norm" not in k and v.ndim >= 2:
                print(k)
                # torch.linalg.svd is faster than the deprecated torch.svd on GPU.
                # full_matrices=False -> reduced SVD: O(min(m,n)^2 * max(m,n)).
                U, S, Vh = torch.linalg.svd(v, full_matrices=False)
                decomposed_params[f"{k}.U"] = U.to(torch.bfloat16).cpu()
                decomposed_params[f"{k}.S"] = S.to(torch.bfloat16).cpu()
                decomposed_params[f"{k}.V"] = Vh.transpose(-2, -1).to(torch.bfloat16).cpu()
        torch.save(decomposed_params, decomposed_param_file)
        print("successfully decomposed model - returning")
        return
    elif extract_svd:
        print(f"ERROR: SVD file already exists at {decomposed_param_file}")
    else:
        print("Decomposed params found. Loading...")
        assert not extract_svd
        decomposed_params = torch.load(decomposed_param_file)
    for k, v in decomposed_params.items():
        decomposed_params[k] = v.to(torch.bfloat16).to(gpu)

    if cfg.wandb_log:
        wandb = wandb_init(
            cfg=cfg, group_name=group_name, run_name=run_name, log_dir=log_dir
        )

    policy: Policy = hydra.utils.instantiate(
        cfg.shakeoff_policy,
        base_params=base_params,
        decomposed_params=decomposed_params,
        gpu=gpu,
    )

    optimization_algorithm: OptimizationAlgorithm = hydra.utils.instantiate(
        cfg.optimization_algorithm,
        policy=policy,
        gpu=gpu,
    )

    # === Baseline eval — vLLM still has original (un-masked) base weights here. ===
    baseline_metrics = {}
    if cfg.get("baseline_eval", True):
        print("=== Baseline eval (base model, no SVD mask applied) ===")
        model.eval()
        if has_training_split:
            b_train = eval_model(vllm_model, train_eval, train_ix)
            b_valid = eval_model(vllm_model, train_eval, valid_ix)
            baseline_metrics["baseline_train_acc"] = b_train.aggregate_metrics[
                task_loader.target_metric_train
            ]
            baseline_metrics["baseline_valid_acc"] = b_valid.aggregate_metrics[
                task_loader.target_metric_valid
            ]
        b_test = eval_model(vllm_model, test_eval)
        baseline_metrics["baseline_test_acc"] = b_test.aggregate_metrics[
            task_loader.target_metric_test
        ]
        if has_transfer_split:
            b_transfer = eval_model(vllm_model, transfer_eval)
            baseline_metrics["baseline_transfer_acc"] = b_transfer.aggregate_metrics[
                task_loader.target_metric_transfer
            ]
        print(f"Baseline metrics: {baseline_metrics}")
        with open(f"{log_dir}/baseline_eval.json", "w") as f:
            json.dump(baseline_metrics, f, indent=4)
        if cfg.wandb_log:
            wandb.log(baseline_metrics)

    if resuming_from_ckpt and os.path.exists(load_ckpt):
        print(f"Starting from checkpoint at: {load_ckpt}")
        # load the lora weight
        if use_lora:
            assert os.path.isdir(load_ckpt), "ckpt for lora must be dir to lora adapter"
            from peft import PeftModel

            lora_model = PeftModel.from_pretrained(model, load_ckpt)
            merged_model = lora_model.merge_and_unload()
            new_params = merged_model.state_dict()
        # load svd expert
        elif "learnable_params" in load_ckpt:
            learnable_params = torch.load(load_ckpt)
            for k, v in learnable_params.items():
                learnable_params[k] = v.to(gpu)
            assert test_only
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )
        else:
            state_dict = torch.load(load_ckpt, weights_only=True)
            policy.load_state_dict(state_dict=state_dict)
            if test_only:
                learnable_params = policy.get_learnable_params()
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )
        load_hf_params_to_vllm(new_params, vllm_model.llm)
    else:
        print(f"Starting from the base model as load_ckpt=={load_ckpt}")

    model.eval()

    # Prompt-based and cls dispatcher evaluation.
    if test_only and prompt_based_eval:
        test_data_dict = eval_model_experts_prompt_based(
            vllm_model,
            test_eval,
            experts_path_dict,
            policy,
            model,
            base_params,
            decomposed_params,
            task_loader.target_metric_test,
        )
        test_data_dict["type"] = "test"
        # Log the results.
        if cfg.wandb_log:
            wandb.log(test_data_dict)
        with open(f"{log_dir}/eval_results.json", "w") as f:
            json.dump(test_data_dict, f, indent=4)
        print(f"Test evaluation results: {test_data_dict}")

        # Eval the transfer set if available
        if has_transfer_split:
            transfer_data_dict = eval_model_experts_prompt_based(
                vllm_model,
                transfer_eval,
                experts_path_dict,
                policy,
                model,
                base_params,
                decomposed_params,
                task_loader.target_metric_transfer,
            )
            transfer_data_dict["type"] = "transfer"
            # Log the results.
            if cfg.wandb_log:
                wandb.log(transfer_data_dict)
            with open(f"{log_dir}/eval_results.json", "w") as f:
                json.dump(transfer_data_dict, f, indent=4)
            print(f"Transfer evaluation results: {transfer_data_dict}")

        return

    # Non-adaptive evaluation on train, val, test set.
    if test_only and not prompt_based_eval:
        data_dict = {}
        details_dict = {}
        if has_training_split:
            train_res = eval_model(vllm_model, train_eval, train_ix)
            valid_res = eval_model(vllm_model, train_eval, valid_ix)
            data_dict["train_acc"] = train_res.aggregate_metrics[
                task_loader.target_metric_train
            ]
            data_dict["valid_acc"] = valid_res.aggregate_metrics[
                task_loader.target_metric_valid
            ]
            details_dict["train"] = train_res.sample_details
            details_dict["valid"] = valid_res.sample_details
        test_res = eval_model(vllm_model, test_eval)
        data_dict["test_acc"] = test_res.aggregate_metrics[
            task_loader.target_metric_test
        ]
        details_dict["test"] = test_res.sample_details
        if has_transfer_split:
            transfer_res = eval_model(vllm_model, transfer_eval)
            data_dict["transfer_acc"] = transfer_res.aggregate_metrics[
                task_loader.target_metric_transfer
            ]
            details_dict["transfer"] = transfer_res.sample_details
        if cfg.wandb_log:
            wandb.log(data_dict)
        with open(f"{log_dir}/eval_results.json", "w") as f:
            json.dump({"baseline": baseline_metrics, "trained": data_dict}, f, indent=4)
        print(f"Evaluation results: {data_dict}")
        if baseline_metrics:
            print("=== Baseline vs Trained ===")
            for split in ("train_acc", "valid_acc", "test_acc"):
                if split in data_dict:
                    base_v = baseline_metrics.get(f"baseline_{split}", None)
                    trained_v = data_dict[split]
                    delta = (
                        f"  Δ={trained_v - base_v:+.4f}"
                        if base_v is not None
                        else ""
                    )
                    print(
                        f"  {split:10s}: base={base_v if base_v is not None else 'n/a'} "
                        f"-> trained={trained_v:.4f}{delta}"
                    )
        return

    learnable_params = policy.get_learnable_params()
    for k in learnable_params:
        model.get_parameter(k).requires_grad_(True)

    # Training loop.
    if batch_size is None:
        clipped_batch_size = len(list(train_ix))
    else:
        clipped_batch_size = min(batch_size, len(list(train_ix)))
    best_val_acc = 0.0
    test_at_best = 0.0
    transfer_at_best = 0.0
    for i in range(num_iters):

        batch_ix = np_random.choice(train_ix, size=clipped_batch_size, replace=False)

        optimization_algorithm.step_optimization(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            policy=policy,
            task_loader=task_loader,
            batch_ix=batch_ix,
            train_data=train_data,
            train_eval=train_eval,
            base_params=base_params,
            decomposed_params=decomposed_params,
            original_model_params=original_model_params,
            metrics_to_log=metrics_to_log,
            vllm_model=vllm_model,
        )

        with torch.no_grad():
            lists_to_log = {}
            grads = [p.grad for p in policy.trainable_params]
            if grads[0] is not None:
                grad_mean = [g.mean().item() for g in grads]
                grad_mags = [torch.linalg.vector_norm(g).item() for g in grads]
                lists_to_log["grad_mean"] = grad_mean
                lists_to_log["grad_mags"] = grad_mags

                param_mags = [
                    torch.linalg.vector_norm(p).item() for p in policy.trainable_params
                ]
                lists_to_log["policy_param_mag"] = param_mags

            generated_params_list = list(learnable_params.values())

            generated_param_mean = [p.mean().item() for p in generated_params_list]
            generated_param_mags = [
                torch.linalg.vector_norm(p).item() for p in generated_params_list
            ]
            lists_to_log["generated_param_mean"] = generated_param_mean
            lists_to_log["generated_param_mags"] = generated_param_mags

            list_stats = {}
            for k, v in lists_to_log.items():
                list_stats.update(get_mean_std_max_min_dict(array=v, prefix=k))
            metrics_to_log.update(**list_stats)

        optimization_algorithm.update(policy=policy)

        # Make sure old params are deleted and garbage-collected
        gc.collect()
        torch.cuda.empty_cache()
        model.zero_grad()

        # More accurate logging.
        value_mean = list_stats.get("generated_param_mean/mean", None)
        grad_mean_mag = list_stats.get("grad_mags/mean", None)
        print(
            f"Iter {i}: "
            + f"param_mean={value_mean}, "
            + f"grad_mean_mag={grad_mean_mag}"
        )
        optimization_algorithm.log_optim(metrics_to_log=metrics_to_log)

        # Test and save.
        if i % test_interval == 0:
            learnable_params = policy.get_learnable_params()
            forward(policy, model, base_params, decomposed_params, learnable_params)
            load_hf_params_to_vllm(model.state_dict(), vllm_model.llm)

            train_res = eval_model(vllm_model, train_eval, train_ix)
            valid_res = eval_model(vllm_model, train_eval, valid_ix)
            test_res = eval_model(vllm_model, test_eval)
            if has_transfer_split:
                transfer_res = eval_model(vllm_model, transfer_eval)
            if (
                valid_res.aggregate_metrics[task_loader.target_metric_valid]
                > best_val_acc
            ):
                best_val_acc = valid_res.aggregate_metrics[
                    task_loader.target_metric_valid
                ]
                test_at_best = test_res.aggregate_metrics[
                    task_loader.target_metric_test
                ]
                if has_transfer_split:
                    transfer_at_best = transfer_res.aggregate_metrics[
                        task_loader.target_metric_transfer
                    ]
                print("best_val_acc updated")
                path = f"{log_dir}/policy_params.pt"
                torch.save(policy.state_dict(), path)
                if save_legacy_params:
                    torch.save(learnable_params, f"{log_dir}/learnable_params.pt")

            path = f"{log_dir}/policy_params_latest.pt"
            torch.save(policy.state_dict(), path)
            if save_legacy_params:
                torch.save(learnable_params, f"{log_dir}/learnable_params_latest.pt")

            policy.record_state(metrics_to_log=metrics_to_log)
            data_dict = {
                "iter": i,
                "best_val_acc": best_val_acc,
                "test_at_best_val": test_at_best,
                "train_acc": train_res.aggregate_metrics[
                    task_loader.target_metric_train
                ],
                "valid_acc": valid_res.aggregate_metrics[
                    task_loader.target_metric_valid
                ],
                "test_acc": test_res.aggregate_metrics[task_loader.target_metric_test],
                **metrics_to_log.get(),
            }
            if has_transfer_split:
                data_dict["transfer_acc"] = transfer_res.aggregate_metrics[
                    task_loader.target_metric_transfer
                ]
                data_dict["transfer_at_best_val"] = transfer_at_best
            print(
                f"Iter {i} eval | "
                f"train_acc={data_dict['train_acc']:.4f} | "
                f"valid_acc={data_dict['valid_acc']:.4f} | "
                f"test_acc={data_dict['test_acc']:.4f} | "
                f"best_val={best_val_acc:.4f} (test_at_best={test_at_best:.4f})"
            )
            if cfg.wandb_log:
                wandb.log(data_dict)
            with open(f"{log_dir}/reinforce_log.json", "a") as f:
                json_data = json.dumps(data_dict, indent=4)
                f.write(json_data)
                f.write("\n")
            metrics_to_log.reset()


if __name__ == "__main__":
    main()
