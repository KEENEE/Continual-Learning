import math

import torch
import torch.nn as nn


class LoRAPolicy(nn.Module):
    requires_svd = False

    def __init__(self, base_params, gpu, rank=16, alpha=32, dropout=0.05, **kwargs):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.decomposable_keys = []
        self.num_params = 0

        for k, v in base_params.items():
            if "norm" not in k and "embed" not in k and v.ndim >= 2:
                safe_key = k.replace(".", "_")
                self.decomposable_keys.append(k)
                out_dim, in_dim = v.shape[0], v.shape[1]

                A = torch.empty(rank, in_dim, device=gpu, dtype=torch.bfloat16)
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                self.lora_A[safe_key] = nn.Parameter(A)

                B = torch.zeros(out_dim, rank, device=gpu, dtype=torch.bfloat16)
                self.lora_B[safe_key] = nn.Parameter(B)

                self.num_params += A.numel() + B.numel()

        print(f"LoRA #params={self.num_params} (rank={rank}, alpha={alpha}, dropout={dropout})")
        self.trainable_params = list(self.lora_A.values()) + list(self.lora_B.values())
        self._key_to_safe = {k: k.replace(".", "_") for k in self.decomposable_keys}
        self._base_weights = {}

    def set_base_weights(self, base_params):
        """Store cloned base weights for composition.

        model.state_dict() returns references to parameter storage;
        forward() overwrites those in-place via .copy_(), so we must
        clone to keep the original base weights intact.
        """
        self._base_weights = {
            k: v.clone() for k, v in base_params.items() if k in self._key_to_safe
        }

    def get_learnable_params(self, detach=False):
        return {k: self.lora_A[self._key_to_safe[k]] for k in self.decomposable_keys}

    def get_mask(self, p):
        return p

    def compose_new_params(self, param_name, decomposed_params, learnable_params):
        """Compose new weight as base_weight + scaling * B @ A.

        Dropout is only applied when training (i.e. during backward pass
        graph construction). During forward/inference the policy should be
        in eval mode so dropout is identity — this ensures the weights used
        for generation match what backward differentiates through.
        """
        safe = self._key_to_safe[param_name]
        A = self.lora_A[safe]
        B = self.lora_B[safe]
        return self._base_weights[param_name] + self.scaling * (B @ self.dropout(A))

    def set_trainable_params_values(self, new_values):
        with torch.no_grad():
            for p, v in zip(self.trainable_params, new_values):
                p.data.copy_(v)

    def record_state(self, metrics_to_log):
        pass
