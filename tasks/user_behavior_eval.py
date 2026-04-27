from dataclasses import dataclass

import fishfarm
import numpy as np


@dataclass
class _Result:
    sample_details: list
    aggregate_metrics: dict


class UserBehaviorEvaluator:
    """Evaluator for the user-behavior task.

    Generates with vLLM, then scores each generation against the gold answer
    using sentence-transformer cosine similarity. A prediction is "correct"
    when cosine >= threshold.
    """

    def __init__(
        self,
        samples,
        system_msg,
        threshold=0.7,
        embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.samples = samples
        self.system_msg = system_msg
        self.threshold = threshold
        self._embed_model_name = embed_model
        self._embedder = None
        self._gold_emb_cache = None  # numpy [N, D]; built lazily after embedder loads

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            # GPU if available — embedder runs in inference-only mode and is small.
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            self._embedder = SentenceTransformer(self._embed_model_name, device=device)

    def _gold_embeddings(self, ids):
        """Cache all gold embeddings once; return rows for `ids`."""
        if self._gold_emb_cache is None:
            self._gold_emb_cache = self._embedder.encode(
                [s.gold_answer for s in self.samples],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        import numpy as _np

        return _np.asarray(self._gold_emb_cache)[ids]

    @staticmethod
    def _clean_generation(g):
        g = g.strip()
        if "Next action:" in g:
            g = g.split("Next action:")[-1].strip()
        return g.splitlines()[0].strip() if g else ""

    def evaluate(self, vllm_model, sample_ids=None):
        self._ensure_embedder()
        ids = list(range(len(self.samples))) if sample_ids is None else list(sample_ids)
        reqs = [
            fishfarm.models.GenerationRequest(
                messages=[
                    fishfarm.Message("system", self.system_msg),
                    fishfarm.Message("user", self.samples[i].input_text),
                ]
            )
            for i in ids
        ]
        outs = vllm_model.generate(reqs)
        gens_raw = [o.generation for o in outs]
        gens_clean = [self._clean_generation(g) for g in gens_raw]
        golds = [self.samples[i].gold_answer for i in ids]

        gen_emb = self._embedder.encode(
            gens_clean, normalize_embeddings=True, show_progress_bar=False
        )
        gold_emb = self._gold_embeddings(ids)
        sims = (np.asarray(gen_emb) * np.asarray(gold_emb)).sum(-1)

        details = [
            {
                "output": gens_raw[k],
                "cleaned": gens_clean[k],
                "gold": golds[k],
                "sim": float(sims[k]),
                "correct": bool(sims[k] >= self.threshold),
            }
            for k in range(len(ids))
        ]
        agg = {
            "sem_acc": float(sum(d["correct"] for d in details) / max(len(details), 1)),
            "sem_sim_mean": float(sims.mean()) if len(sims) else 0.0,
        }
        return _Result(sample_details=details, aggregate_metrics=agg)
