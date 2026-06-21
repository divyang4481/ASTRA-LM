import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from .config import ModelConfig

class VayuSphereAdapter(nn.Module):
    """
    VayuSphere Adapter: A lightweight angular Q/K gate that preserves Flash/SDPA compatibility.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.alpha = config.vayusphere_alpha
        self.temperature = config.vayusphere_temperature
        self.target = config.vayusphere_target
        self.normalize = config.vayusphere_normalize

        if config.use_vayusphere:
            self.centroids = nn.Parameter(
                torch.randn(config.vayusphere_num_centroids, config.head_dim) * 0.02
            )
        else:
            self.register_parameter("centroids", None)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            q: [B, H, S, Hd]
            k: [B, Hkv, S, Hd]
        """
        diagnostics = {}

        if not self.config.use_vayusphere or self.alpha == 0:
            return q, k, diagnostics

        # Normalize directions on hypersphere if enabled
        if self.normalize:
            q_norm = F.normalize(q, dim=-1, eps=1e-6)
            k_norm = F.normalize(k, dim=-1, eps=1e-6)
            c_norm = F.normalize(self.centroids, dim=-1, eps=1e-6)
        else:
            q_norm = q
            k_norm = k
            c_norm = self.centroids

        if "q" in self.target:
            # sim: [B, H, S, C]
            sim_q = torch.einsum("bhsd,cd->bhsc", q_norm, c_norm)
            weights_q = torch.softmax(sim_q / self.temperature, dim=-1)
            score_q = (weights_q * sim_q).sum(dim=-1, keepdim=True)

            gate_q = torch.tanh(score_q)
            scale_q = 1.0 + self.alpha * gate_q
            scale_q = scale_q.clamp(0.5, 1.5)
            q = q * scale_q

            if return_diagnostics:
                diagnostics["vayusphere_q_gate_mean"] = gate_q.mean().item()
                diagnostics["vayusphere_q_gate_std"] = gate_q.std().item()
                diagnostics["vayusphere_q_scale_mean"] = scale_q.mean().item()
                diagnostics["vayusphere_q_scale_std"] = scale_q.std().item()

        if "k" in self.target:
            # sim: [B, Hkv, S, C]
            sim_k = torch.einsum("bhsd,cd->bhsc", k_norm, c_norm)
            weights_k = torch.softmax(sim_k / self.temperature, dim=-1)
            score_k = (weights_k * sim_k).sum(dim=-1, keepdim=True)

            gate_k = torch.tanh(score_k)
            scale_k = 1.0 + self.alpha * gate_k
            scale_k = scale_k.clamp(0.5, 1.5)
            k = k * scale_k

            if return_diagnostics:
                diagnostics["vayusphere_k_gate_mean"] = gate_k.mean().item()
                diagnostics["vayusphere_k_gate_std"] = gate_k.std().item()
                diagnostics["vayusphere_k_scale_mean"] = scale_k.mean().item()
                diagnostics["vayusphere_k_scale_std"] = scale_k.std().item()

        if return_diagnostics:
            c_norm = torch.norm(self.centroids, dim=-1)
            diagnostics["vayusphere_centroid_norm_mean"] = c_norm.mean().item()
            diagnostics["vayusphere_centroid_norm_std"] = c_norm.std().item()

        return q, k, diagnostics
