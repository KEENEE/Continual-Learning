import fishfarm
import vllm
from datasets import load_dataset
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.ai2_arc import Ai2ArcSample, Ai2ArcTask

from .base import LLAMA3_COT, Task, freeze_vllm_model_grads, get_download_dir

choices = ["A", "B", "C", "D", "E"]


class AI2ArcTask(Task):
    def __init__(
        self,
    ):
        self.model_to_template = {
            "meta-llama/Meta-Llama-3-8B-Instruct": LLAMA3_COT,
            "mistralai/Mistral-7B-Instruct-v0.3": None,
        }
        self.system_msg = (
            "The following are multiple choice questions (with answers). "
            "Think step by step and then finish your answer "
            'with "the answer is (X)" where X is the correct letter choice.'
        )
        self.target_metric_train = "acc"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = True
        self.has_training_split = True

    def get_train_data(
        self,
    ):
        train_eval, *test_evals = self.get_evaluator()
        train_data = train_eval.samples
        train_size = len(train_data)
        train_ix = range(0, train_size, 2)
        valid_ix = range(1, train_size, 2)
        return train_data, train_ix, valid_ix

    def get_rewards(self, res):
        rewards = [1.0 if x["correct"] else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(
        self,
    ):
        res = []
        for split in ["train", "test"]:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
            samples = []
            for sample in dataset:
                options = []
                for opt in sample["choices"]["text"]:
                    options.append(opt)
                # add options to the question
                question = sample["question"] + "\n"
                question += "Options:\n"
                for i, opt in enumerate(options):
                    question += "{}. {}\n".format(choices[i], opt)
                samples.append(
                    Ai2ArcSample(
                        question=question,
                        answer=sample["answerKey"],
                        options=options,
                        question_id=sample["id"],
                    )
                )
            res.append(
                Ai2ArcTask(
                    samples=samples,
                    context_messages=[
                        fishfarm.Message("system", self.system_msg),
                    ],
                )
            )
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        samples = []
        for sample in dataset:
            options = []
            for opt in sample["choices"]["text"]:
                options.append(opt)
            # add options to the question
            question = sample["question"] + "\n"
            question += "Options:\n"
            for i, opt in enumerate(options):
                question += "{}. {}\n".format(choices[i], opt)
            samples.append(
                Ai2ArcSample(
                    question=question,
                    answer=sample["answerKey"],
                    options=options,
                    question_id=sample["id"],
                )
            )
        res.append(
            Ai2ArcTask(
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
