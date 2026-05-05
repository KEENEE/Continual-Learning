import json
import os
from dataclasses import dataclass
from typing import Optional

import vllm
from fishfarm.models.vllm_model import VLLMModel

from .base import LLAMA3_COT, Task, freeze_vllm_model_grads, get_download_dir
from .user_behavior_eval import UserBehaviorEvaluator

DEFAULT_SAMPLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "user_behavior_samples.json",
)

SYSTEM_MSG = (
    "You are a personal assistant that predicts a user's next action.\n"
    "Given the user's recent usage history and a current "
    "trigger event, output the single most likely next action as a short phrase.\n"
    "Reply with the action only — no explanation."
)

@dataclass
class UserBehaviorSample:
    input_text: str
    gold_answer: str
    gold_reasoning: str

    # Provide a `question` field so utils.classify_samples (if ever invoked)
    # has a consistent attribute to read.
    @property
    def question(self):
        return self.input_text


class UserBehaviorTask(Task):
    def __init__(
        self,
        samples_path: str = DEFAULT_SAMPLES_PATH,
        use_reasoning_in_target: bool = False,
        embed_threshold: float = 0.7,
        embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_model_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        # Chat template: same Llama3-style template for the two Llama models;
        # Gemma falls back to the tokenizer default.
        llama_tpl = LLAMA3_COT.replace("Let\\'s think step by step", "")
        self.model_to_template = {
            "meta-llama/Meta-Llama-3-8B-Instruct": llama_tpl,
            "meta-llama/Meta-Llama-3.1-8B-Instruct": llama_tpl,
            "google/gemma-4-E4B": None,
        }
        self.system_msg = SYSTEM_MSG
        self.target_metric_train = "sem_acc"
        self.target_metric_valid = "sem_acc"
        self.target_metric_test = "sem_acc"
        self.target_metric_transfer = "sem_acc"
        self.has_transfer_split = False
        self.has_training_split = True

        self.use_reasoning_in_target = use_reasoning_in_target
        self.embed_threshold = embed_threshold
        self.embed_model_name = embed_model
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens

        self.train_samples, self.valid_samples, self.test_samples = self._load(
            samples_path
        )

    @staticmethod
    def _load(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pre-built samples not found at {path}. "
                f"Run: python scripts/build_user_behavior_samples.py"
            )
        with open(path) as f:
            data = json.load(f)
        to_samples = lambda items: [UserBehaviorSample(**s) for s in items]
        return (
            to_samples(data["train"]),
            to_samples(data["valid"]),
            to_samples(data["test"]),
        )

    # ----- Task interface -----

    def get_train_data(self):
        train_eval_samples = self.train_samples + self.valid_samples
        n_train = len(self.train_samples)
        n_valid = len(self.valid_samples)
        train_ix = list(range(n_train))
        valid_ix = list(range(n_train, n_train + n_valid))
        return train_eval_samples, train_ix, valid_ix

    def get_evaluator(self):
        train_eval = UserBehaviorEvaluator(
            samples=self.train_samples + self.valid_samples,
            system_msg=self.system_msg,
            threshold=self.embed_threshold,
            embed_model=self.embed_model_name,
        )
        test_eval = UserBehaviorEvaluator(
            samples=self.test_samples,
            system_msg=self.system_msg,
            threshold=self.embed_threshold,
            embed_model=self.embed_model_name,
        )
        return (train_eval, test_eval)

    def get_rewards(self, res):
        return [float(2 * x["sim"] - 1) for x in res.sample_details]

    def get_prompt(self, tokenizer, samples, ix, model_id):
        s = samples[ix]
        chat_template = self.model_to_template.get(model_id)
        return tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": self.system_msg},
                {"role": "user", "content": s.input_text},
            ],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_gold_target(self, sample):
        if self.use_reasoning_in_target and sample.gold_reasoning:
            return f"{sample.gold_reasoning}\nNext action: {sample.gold_answer}"
        return sample.gold_answer

    def get_vllm_model(self, model_id, tensor_parallel_size=1):
        max_model_len = self.max_model_len or (
            8192 if "Llama-3-8B" in model_id else 16384
        )
        max_tokens = self.max_tokens or (256 if self.use_reasoning_in_target else 64)
        # Gemma 4 checkpoints declare Gemma4ForConditionalGeneration (multimodal)
        # in config.json. Force the text-only Gemma4ForCausalLM path so vLLM does
        # not load the vision/audio encoders.
        llm_kwargs = dict(
            max_model_len=max_model_len,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            download_dir=get_download_dir(),
        )
        if "gemma-4" in model_id.lower():
            llm_kwargs["hf_overrides"] = {"architectures": ["Gemma4ForCausalLM"]}
        model = vllm.LLM(model_id, **llm_kwargs)
        chat_template = self.model_to_template.get(model_id)
        freeze_vllm_model_grads(model)
        vllm_model = VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=max_tokens,
                stop=["\n\n", "<|eot_id|>", "</s>"],
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
        )
        return vllm_model
