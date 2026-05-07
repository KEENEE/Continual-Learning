import re
from copy import deepcopy
from typing import Dict, Optional

import fishfarm
import torch
import torch.utils
import vllm


def _try_get_parameter(model, name: str):
    """Return model.get_parameter(name) or None if the parameter is absent.

    Some checkpoints (e.g. Gemma 4 with tie_word_embeddings=True) don't
    expose lm_head as its own parameter, so callers must skip silently.
    """
    try:
        return model.get_parameter(name)
    except AttributeError:
        return None


def _copy_weights(model, param: Dict) -> None:
    """Copy SVD-decomposed weights from a (possibly partial) HF
    state-dict into a live vLLM nn.Module.

    Handles the packing conventions used by vLLM:
      - q/k/v_proj → qkv_proj   (attention)
      - gate/up_proj → gate_up_proj  (MLP)
    Tolerates partial dicts so callers can ship one layer per IPC call.
    """
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    num_layers = text_config.num_hidden_layers

    # lm_head (may be tied with embed_tokens and absent)
    if "lm_head.weight" in param:
        lm_head = _try_get_parameter(model, "lm_head.weight")
        if lm_head is not None:
            lm_head.copy_(
                param["lm_head.weight"][: lm_head.shape[0]]
                .to(lm_head.dtype)
                .to(lm_head.device)
            )

    for i in range(num_layers):
        # --- Attention ---
        q_key = f"model.layers.{i}.self_attn.q_proj.weight"
        k_key = f"model.layers.{i}.self_attn.k_proj.weight"
        v_key = f"model.layers.{i}.self_attn.v_proj.weight"
        if q_key in param and k_key in param and v_key in param:
            qkv = model.get_parameter(
                f"model.layers.{i}.self_attn.qkv_proj.weight"
            )
            qkv.copy_(
                torch.cat(
                    [param[q_key], param[k_key], param[v_key]], dim=0
                )
                .to(qkv.dtype)
                .to(qkv.device)
            )
        o_key = f"model.layers.{i}.self_attn.o_proj.weight"
        if o_key in param:
            o = model.get_parameter(o_key)
            o.copy_(param[o_key].to(o.dtype).to(o.device))

        # --- MLP ---
        gate_key = f"model.layers.{i}.mlp.gate_proj.weight"
        up_key = f"model.layers.{i}.mlp.up_proj.weight"
        if gate_key in param and up_key in param:
            gate_up = model.get_parameter(
                f"model.layers.{i}.mlp.gate_up_proj.weight"
            )
            gate_up.copy_(
                torch.cat([param[gate_key], param[up_key]], dim=0)
                .to(gate_up.dtype)
                .to(gate_up.device)
            )
        down_key = f"model.layers.{i}.mlp.down_proj.weight"
        if down_key in param:
            down = model.get_parameter(down_key)
            down.copy_(param[down_key].to(down.dtype).to(down.device))


def load_hf_params_to_vllm(param: Dict, llm: vllm.LLM) -> None:
    """Ship updated weights into the live vLLM engine."""
    # Filter to SVD-decomposed weights only and (for gemma 4) strip the
    # multimodal model.language_model.* prefix so keys match vLLM's
    # text-only Gemma4ForCausalLM layout.
    is_gemma4 = any("language_model" in k for k in param)
    if is_gemma4:
        prefix = "model.language_model."
        filtered: Dict = {}
        for k, v in param.items():
            if any(s in k for s in (
                "vision_tower", "audio_tower", "embed_vision", "embed_audio",
            )):
                continue
            if "norm" in k or "embed" in k or v.ndim < 2:
                continue
            if k.startswith(prefix):
                k = "model." + k[len(prefix):]
            filtered[k] = v
        param = filtered
    else:
        param = {k: v for k, v in param.items()
                 if "norm" not in k and "embed" not in k and v.ndim >= 2}

    if not hasattr(llm, "apply_model"):
        # vLLM v0 path — in-process, no IPC, no size cap.
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        _copy_weights(model, param)
        return

    # vLLM v1: apply_model ships its callback through an IPC channel
    # whose msgspec encoder caps any single object at 2**32 - 1 bytes.
    # Send one layer at a time; non-layer weights (e.g. lm_head) go in
    # a separate chunk.
    layer_re = re.compile(r"model\.layers\.(\d+)\.")
    by_layer: Dict[int, Dict] = {}
    non_layer: Dict = {}
    for k, v in param.items():
        m = layer_re.search(k)
        if m:
            by_layer.setdefault(int(m.group(1)), {})[k] = v
        else:
            non_layer[k] = v
    if non_layer:
        llm.apply_model(lambda m, p=non_layer: _copy_weights(m, p))
    for chunk in by_layer.values():
        llm.apply_model(lambda m, p=chunk: _copy_weights(m, p))


def eval_model(vllm_model, evaluator, ix=None):
    result = evaluator.evaluate(vllm_model, sample_ids=ix)
    return result


def compose_new_params(
    policy,
    param_name,
    decomposed_params,
    learnable_params,
):
    """Compose new parameters — delegates to policy.compose_new_params."""
    return policy.compose_new_params(param_name, decomposed_params, learnable_params)


@torch.no_grad()
def forward(policy, model, base_params, decomposed_params, learnable_params):
    """Forward pass."""
    new_params = {}
    for k, v in base_params.items():
        if "norm" not in k and "embed" not in k and v.ndim >= 2:
            new_params[k] = compose_new_params(
                policy, k, decomposed_params, learnable_params
            )
            model.get_parameter(k).copy_(new_params[k])
        else:
            new_params[k] = base_params[k]
    return new_params


@torch.no_grad()
def load_base_params(
    model,
    base_params,
):
    for k, v in base_params.items():
        if "norm" not in k and "embed" not in k and v.ndim >= 2:
            model.get_parameter(k).copy_(v.cuda())


def backward(
    policy,
    model,
    base_params,
    decomposed_params,
    learnable_params,
):
    """Backward pass."""
    keys_to_backprop = [k for k, v in base_params.items() if "norm" not in k and "embed" not in k and v.ndim >= 2]
    last_key = keys_to_backprop[-1]
    for k in keys_to_backprop[:-1]:
        compose_new_params(policy, k, decomposed_params, learnable_params).backward(
            model.get_parameter(k).grad, retain_graph=True
        )
    # release graph
    compose_new_params(policy, last_key, decomposed_params, learnable_params).backward(
        model.get_parameter(last_key).grad, retain_graph=False
    )


def classify_samples(vllm_model, test_eval):
    """Classify samples."""

    CLASSIFICATION_PROMPT = """
    # Analyze the given question and classify it into one of four categories: 'code', 'math', 'reasoning' or 'other'. Follow these guidelines:

    1. Code: Questions asking for programming solutions, functions, algorithms. Often includes specific programming terms, language syntax, or data structures.
    2. Math: Questions involving mathematical calculations, formulas, statistics. Often includes numbers, equations, or mathematical operations.
    3. Reasoning: Questions requiring logical thinking, application of scientific knowledge, or critical analysis of information. Often presents statements that need evaluation based on general understanding. 
    4. Other: Questions not clearly fit into above categories.

    Instructions:
    - Consider the primary focus, skills, and knowledge required to answer the question.
    - If a question spans multiple categories, choose the most dominant one.
    - Provide your final classification within \\boxed{} notation. Example: \\boxed{reasoning}

    Format your response as follows:
    Classification: \\boxed{category}
    """

    def extract_classification(text: str) -> Optional[str]:
        """
        Extract the classification from the model's output using regex.
        """
        match = re.search(r"\\boxed{([^}]*)}", text)
        return match.group(1) if match else None

    # Identify the key in the samples that contains the problem text
    problem_key = None
    for key in ("problem", "question", "instruction"):
        if (
            hasattr(test_eval.samples[0], key)
            and getattr(test_eval.samples[0], key) is not None
        ):
            problem_key = key
            break
    assert problem_key is not None, "Could not find problem text in the samples"

    # Prepare classification requests
    classification_requests = [
        fishfarm.models.GenerationRequest(
            messages=[
                fishfarm.Message("system", CLASSIFICATION_PROMPT),
                fishfarm.Message("user", getattr(sample, problem_key)),
            ]
        )
        for sample in test_eval.samples
    ]

    # Generate classifications using the model
    model_outputs = vllm_model.generate(classification_requests)

    # Process results and update samples
    classified_samples = []
    for sample, result in zip(test_eval.samples, model_outputs):
        prediction = extract_classification(result.generation)
        if prediction not in ["code", "math", "reasoning"]:
            prediction = "other"
        sample.expert_label = prediction
        classified_samples.append(sample)

    return classified_samples


def eval_model_experts_prompt_based(
    vllm_model,
    evaluator,
    experts_path_dict,
    policy,
    model,
    base_params,
    decomposed_params,
    task_metric,
):
    """Evaluate the model using expert models and prompt-based classification."""
    results_by_expert: Dict[str, Dict] = {}

    # Classify all test samples
    classified_samples = classify_samples(vllm_model, evaluator)

    # Evaluate samples for each expert model
    for expert_label, expert_model_path in experts_path_dict.items():
        # Filter samples for current expert
        expert_samples = [
            sample
            for sample in classified_samples
            if sample.expert_label == expert_label
        ]
        if not expert_samples:
            continue

        # Update test evaluation with filtered samples
        evaluator.samples = expert_samples

        # Load and apply expert model parameters if available
        if expert_model_path:
            policy.load_state_dict(torch.load(expert_model_path))
            expert_params = policy.get_learnable_params()
            updated_params = forward(
                policy=policy,
                model=model,
                base_params=base_params,
                decomposed_params=decomposed_params,
                learnable_params=expert_params,
            )
            load_hf_params_to_vllm(updated_params, vllm_model.llm)

        # Evaluate current expert model
        evaluation_results = eval_model(vllm_model, evaluator)

        # Store results for current expert
        results_by_expert[expert_label] = {
            "num_samples": len(expert_samples),
            "test_acc": evaluation_results.aggregate_metrics[task_metric],
        }

    # Compute the overall accuracy.
    data_dict = deepcopy(results_by_expert)
    data_dict["final_test_acc"] = 0.0
    for label in results_by_expert.keys():
        data_dict["final_test_acc"] += (
            results_by_expert[label]["test_acc"]
            * results_by_expert[label]["num_samples"]
        )
    data_dict["final_test_acc"] /= len(classified_samples)

    return data_dict
