import re
from dataclasses import dataclass
from typing import Iterable, Tuple

import datasets
import fishfarm
import vllm
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.base import TaskResult
from fishfarm.tasks.evalplus import load_dataset

from .base import Task, freeze_vllm_model_grads, get_download_dir


def mean(iterable: Iterable[float]) -> float:
    total, count = 0.0, 0
    for x in iterable:
        total += x
        count += 1
    return total / count


def extract_ans(text):
    """Fetch the string within \\boxed{}."""
    match = re.search(r"\\boxed{([^}]*)}", text)
    if match:
        return match.group(1)  # Return the content inside the \boxed{}
    else:
        return None  # Return None if no match is found


@dataclass
class CategorySample:
    question: str
    label: str


class CategoryClassficiationTask(fishfarm.tasks.base.Task):
    def __init__(
        self,
        samples,
        context_messages,
    ):
        self.samples = list(samples)
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model,
        sample_ids,
    ):
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(fishfarm.Message(role="user", content=sample.question))
            requests.append(fishfarm.models.GenerationRequest(messages=messages))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation
            prediction = extract_ans(output)

            sample_details.append(
                dict(
                    question=sample.question,
                    label=sample.label,
                    output=output,
                    prediction=prediction,
                    correct=sample.label == prediction,
                )
            )

        aggregate_metrics = {
            "acc": mean(
                float(sd["correct"]) if isinstance(sd["correct"], (bool)) else 0.0
                for sd in sample_details
            )
        }
        return TaskResult(
            aggregate_metrics=aggregate_metrics, sample_details=sample_details
        )


class ClsTask(Task):
    def __init__(self):
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
            "google/gemma-4-E4B": None,
        }
        self.system_msg = """
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
        self.target_metric_train = "acc"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = False
        self.has_training_split = True
        self.num_samples_per_task = 400  # Hard code 400 samples per task
        self.task_datasets = [
            datasets.load_dataset("gsm8k", "main", split="test"),
            load_dataset(source_dataset="mbpp"),
            datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"),
        ]
        self.train_samples, self.test_samples = self.build_samples()

    def split_samples(self, samples):
        """Split samples into train and test sets with rate 4 : 1."""
        train_samples = []
        test_samples = []
        for i, sample in enumerate(samples):
            if i % 5 < 4:
                train_samples.append(sample)
            else:
                test_samples.append(sample)
        return train_samples, test_samples

    def get_train_data(self=400):
        train_ix = range(0, len(self.train_samples), 2)
        valid_ix = range(1, len(self.train_samples), 2)

        return self.train_samples, train_ix, valid_ix

    def build_samples(self):
        task_labels = ["math", "code", "reasoning"]

        samples = []
        choices = ["A", "B", "C", "D", "E"]
        for dataset, label in zip(self.task_datasets, task_labels):
            counter = 0
            for sample in dataset:
                counter += 1
                if counter >= self.num_samples_per_task:
                    break
                if label == "math":
                    samples.append(
                        CategorySample(
                            question=sample["question"],
                            label="math",
                        )
                    )
                elif label == "code":
                    samples.append(
                        CategorySample(
                            question=sample.instruction,
                            label="code",
                        )
                    )
                else:  # reasoning
                    question = sample["question"] + "\n"
                    question += "Options:\n"
                    options = []
                    for opt in sample["choices"]["text"]:
                        options.append(opt)
                    for i, opt in enumerate(options):
                        question += "{}. {}\n".format(choices[i], opt)
                    samples.append(
                        CategorySample(
                            question=question,
                            label="reasoning",
                        )
                    )

        return self.split_samples(samples)

    def get_rewards(self, res):
        rewards = [1.0 if x["correct"] else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(self) -> Tuple:
        # Build cls dataset here with training tasks.
        res = []
        for samples in [self.train_samples, self.test_samples]:
            res.append(
                CategoryClassficiationTask(
                    samples=samples,
                    context_messages=[
                        fishfarm.Message("system", self.system_msg),
                    ],
                )
            )

        return tuple(res)

    def get_prompt(self, tokenizer, samples, ix, model_id):
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": self.system_msg}
        user_msg = {"role": "user", "content": samples[ix].question}
        prompt = tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def get_vllm_model(self, model_id, tensor_parallel_size=1) -> VLLMModel:
        """Load a vLLM model."""
        llm_kwargs = dict(
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            download_dir=get_download_dir(),
        )
        if "gemma-4" in model_id.lower():
            llm_kwargs["hf_overrides"] = {"architectures": ["Gemma4ForCausalLM"]}
        model = vllm.LLM(model_id, **llm_kwargs)
        chat_template = self.model_to_template[model_id]
        freeze_vllm_model_grads(model)
        vllm_model = VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=1024,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
        )
        return vllm_model
