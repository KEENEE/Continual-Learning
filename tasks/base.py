import os
from abc import ABC, abstractmethod
from typing import Union

import hydra
from omegaconf import DictConfig

LLAMA3_COT = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>"
    "\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + 'Let\\'s think step by step' }}"
    "{% endif %}"
)


CODE_PROMPT = r"""
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n\n' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}

{{ system_message }}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception(
            'Conversation roles must alternate user/assistant/user/assistant/...')}}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '@@ Instruction:\n' + message['content'].strip() + '\n\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '@@ Response:\n' + message['content'].strip() }}
    {% endif %}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '@@ Response:' }}
    {% endif %}
{% endfor %}
""".replace(
    "    ", ""
).replace(
    "\n", ""
)


def get_download_dir():
    if "HF_HOME" in os.environ:
        return os.environ["HF_HOME"] + "/models"
    else:
        return os.path.expanduser("~") + "/.cache/huggingface/models"


def freeze_vllm_model_grads(llm):
    """Set requires_grad=False on every parameter of the underlying vLLM model.

    Works on both legacy V0 (driver_worker.model_runner.model) and V1
    (apply_model) engine layouts so existing callers keep working after
    the dependency bump.
    """

    def _freeze(model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    if hasattr(llm, "apply_model"):
        llm.apply_model(_freeze)
        return
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    _freeze(model)


class Task(ABC):
    def __init__(
        self,
    ):
        self.model_to_template = {}
        self.system_msg = ()
        self.target_metric_train = None
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = None
        self.has_transfer_split = True
        self.has_training_split = True

    @abstractmethod
    def get_train_data(
        self,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_rewards(self, res):
        raise NotImplementedError

    @abstractmethod
    def get_evaluator(
        self,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self, tokenizer, samples, ix, model_id):
        raise NotImplementedError

    @abstractmethod
    def get_vllm_model(self, model_id, tensor_parallel_size=1):
        raise NotImplementedError


class FewShotTask(Task):
    def __init__(
        self,
        wrapped_task: Union[Task, DictConfig],
        wrapped_split: str = "test",
        shots=5,
        seed=16,
    ):
        if isinstance(wrapped_task, Task):
            self.wrapped_task: Task = wrapped_task
        else:
            self.wrapped_task: Task = hydra.utils.instantiate(wrapped_task)

        self.wrapped_split = wrapped_split
        self.shots = shots
        self.seed = seed
        self.model_to_template = wrapped_task.model_to_template
        self.system_msg = wrapped_task.system_msg
        if wrapped_split == "train":
            self.target_metric_train = wrapped_task.target_metric_train
            self.target_metric_valid = wrapped_task.target_metric_train
            self.target_metric_test = wrapped_task.target_metric_train
            assert wrapped_task.has_training_split
        elif wrapped_split == "test":
            self.target_metric_train = wrapped_task.target_metric_test
            self.target_metric_valid = wrapped_task.target_metric_test
            self.target_metric_test = wrapped_task.target_metric_test
        elif wrapped_split == "transfer":
            self.target_metric_train = wrapped_task.target_metric_transfer
            self.target_metric_valid = wrapped_task.target_metric_transfer
            self.target_metric_test = wrapped_task.target_metric_transfer
            assert wrapped_task.has_transfer_split
        else:
            raise NotImplementedError
        self.target_metric_transfer = wrapped_task.target_metric_transfer
        self.has_transfer_split = False
        self.has_training_split = True

    def get_train_data(
        self,
    ):
        train_eval, *test_evals = self.get_evaluator()
        train_data = train_eval.samples
        train_size = len(train_data)
        total_ix = list(range(train_size))
        import random

        random.seed(self.seed)  # fix random seed for reproducibility
        random.shuffle(total_ix)
        train_ix = total_ix[: self.shots]
        valid_ix = total_ix[self.shots :]
        return train_data, train_ix, valid_ix

    def get_rewards(self, res):
        return self.wrapped_task.get_rewards(res=res)

    def get_evaluator(
        self,
    ):
        evaluators = self.wrapped_task.get_evaluator()
        if self.wrapped_split == "train":
            target_eval = evaluators[0]
        elif self.wrapped_split == "test":
            target_eval = evaluators[1]
        elif self.wrapped_split == "transfer":
            target_eval = evaluators[2]
        return target_eval, target_eval

    def get_prompt(self, tokenizer, samples, ix, model_id):
        return self.wrapped_task.get_prompt(
            tokenizer=tokenizer,
            samples=samples,
            ix=ix,
            model_id=model_id,
        )

    def get_vllm_model(self, model_id, tensor_parallel_size=1):
        return self.wrapped_task.get_vllm_model(
            model_id=model_id,
            tensor_parallel_size=tensor_parallel_size,
        )
