from typing import Tuple

import fishfarm
import vllm
from datasets import load_dataset
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.language_restricted_math import (
    LanguageRestrictedMathTask, MathSample, extract_answer_number)

from .base import Task, freeze_vllm_model_grads, get_download_dir


class Gsm8kTask(Task):
    def __init__(
        self,
    ):
        self.model_to_template = {
            "meta-llama/Meta-Llama-3-8B-Instruct": (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>"
                "\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}{% set content = bos_token + content %}"
                "{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            ),
            "mistralai/Mistral-7B-Instruct-v0.3": None,
        }
        self.system_msg = (
            "Below is an instruction that describes a task."
            " Write a response that appropriately completes the request.\n\n"
        )

        self.target_metric_train = "acc"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = False
        self.has_training_split = True

    def get_train_data(
        self,
    ):
        train_data = load_dataset("gsm8k", "main", split="train")
        train_size = len(train_data)
        train_ix = range(0, train_size, 2)
        valid_ix = range(1, train_size, 2)
        return train_data, train_ix, valid_ix

    def get_rewards(self, res):
        rewards = [1.0 if x["correct"] else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(self) -> Tuple:
        res = []
        for split in ["train", "test"]:
            dataset = load_dataset("gsm8k", "main", split=split)
            samples = []
            for sample in dataset:
                answer = sample["answer"]
                answer = extract_answer_number(answer)
                answer = int(answer) if answer is not None else None
                samples.append(
                    MathSample(
                        problem=sample["question"],
                        answer=answer,
                    )
                )
            res.append(
                LanguageRestrictedMathTask(
                    samples=samples,
                    context_messages=[
                        fishfarm.Message("system", self.system_msg),
                    ],
                    languages=[],
                )
            )
        return tuple(res)

    def get_prompt(self, tokenizer, samples, ix, model_id):
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": self.system_msg}
        user_msg = {"role": "user", "content": samples["question"][ix]}
        prompt = tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def get_vllm_model(self, model_id, tensor_parallel_size=1) -> VLLMModel:
        """Load a vLLM model."""
        model = vllm.LLM(
            model_id,
            max_model_len=1024,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            download_dir=get_download_dir(),
        )
        chat_template = self.model_to_template[model_id]
        freeze_vllm_model_grads(model)
        vllm_model = VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=512,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
        )
        return vllm_model
