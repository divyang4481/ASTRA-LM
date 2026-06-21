import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, Dict, Any
from .config import ModelConfig

class VayuSphereAdapter(nn.Module):
    """
    VayuSphere Adapter: A lightweight angular Q/K gate that preserves Flash/SDPA compatibility.
    v0.2: Supports tangent correction, top-k selection, and temperature scheduling.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.alpha = config.vayusphere_alpha
        self.scale_alpha = config.vayusphere_scale_alpha
        self.temperature = config.vayusphere_temperature
        self.target = config.vayusphere_target
        self.normalize = config.vayusphere_normalize
        self.mode = config.vayusphere_mode
        self.topk = config.vayusphere_topk_centroids
        self.current_step = 0

        if config.use_vayusphere:
            self.centroids = nn.Parameter(
                torch.randn(config.vayusphere_num_centroids, config.head_dim) * 0.02,
                requires_grad=not config.vayusphere_freeze_centroids
            )
        else:
            self.register_parameter("centroids", None)

    def update_temperature(self, global_step: int):
        self.current_step = global_step
        if self.config.vayusphere_temperature_decay_steps > 0:
            progress = min(global_step / self.config.vayusphere_temperature_decay_steps, 1.0)
            self.temperature = self.config.vayusphere_temperature_start - progress * (
                self.config.vayusphere_temperature_start - self.config.vayusphere_temperature_min
            )
            self.temperature = max(self.config.vayusphere_temperature_min, self.temperature)

    def _apply_gate(self, x: torch.Tensor, c_norm: torch.Tensor, return_diagnostics: bool = False, prefix: str = "q") -> Tuple[torch.Tensor, Dict[str, Any]]:
        # x: [B, H, S, Hd]
        # c_norm: [C, Hd]
        diag = {}

        # 1. Similarity
        x_hat_for_sim = F.normalize(x, dim=-1, eps=1e-6) if self.normalize else x
        # sim: [B, H, S, C]
        sim = torch.einsum("bhsd,cd->bhsc", x_hat_for_sim, c_norm)

        # 2. Top-k selection
        if 0 < self.topk < self.config.vayusphere_num_centroids:
            topv, topi = torch.topk(sim, k=self.topk, dim=-1)
            sim_masked = torch.full_like(sim, float('-inf'))
            sim_masked.scatter_(-1, topi, topv)
            sim_for_softmax = sim_masked
        else:
            sim_for_softmax = sim

        # 3. Weights and Mixture
        weights = torch.softmax(sim_for_softmax / self.temperature, dim=-1)
        # c_mix: [B, H, S, Hd]
        c_mix = torch.einsum("bhsc,cd->bhsd", weights, c_norm)

        # 4. Score for scaling
        score = (weights * sim).sum(dim=-1, keepdim=True)

        x_out = x

        # 5. Tangent Correction
        if "tangent" in self.mode:
            if self.alpha != 0:
                x_norm_mag = x.norm(dim=-1, keepdim=True).detach()
                x_hat = F.normalize(x, dim=-1, eps=1e-6)
                c_mix_norm = F.normalize(c_mix, dim=-1, eps=1e-6)

                parallel = (c_mix_norm * x_hat).sum(dim=-1, keepdim=True) * x_hat
                tangent = c_mix_norm - parallel
                tangent = F.normalize(tangent, dim=-1, eps=1e-6)

                x_out = x_out + self.alpha * x_norm_mag * tangent

        # 6. Scale Modulation
        if "scale" in self.mode:
            if self.scale_alpha != 0 or (self.mode == "scale" and self.alpha != 0):
                # Backwards compatibility: if mode is "scale", use self.alpha if scale_alpha is 0
                s_alpha = self.scale_alpha if self.mode == "tangent_scale" else self.alpha

                # Centered scale
                centered_score = score - score.mean(dim=-2, keepdim=True)
                gate = torch.tanh(centered_score)
                scale = torch.exp(s_alpha * gate)
                # Original implementation had clamp(0.5, 1.5) and 1.0 + alpha * gate
                # but user asked for exp(alpha * tanh(centered_score))
                x_out = x_out * scale

                if return_diagnostics:
                    diag[f"vayusphere_{prefix}_gate_mean"] = gate.mean().item()
                    diag[f"vayusphere_{prefix}_gate_std"] = gate.std().item()
                    diag[f"vayusphere_{prefix}_gate_min"] = gate.min().item()
                    diag[f"vayusphere_{prefix}_gate_max"] = gate.max().item()

        # 7. Diagnostics
        if return_diagnostics:
            if self.config.vayusphere_enable_heavy_diagnostics:
                # Assignment confidence
                conf = weights.max(dim=-1)[0].mean().item()
                diag[f"vayusphere_{prefix}_assignment_confidence"] = conf

                # Assignment margin
                if self.config.vayusphere_num_centroids > 1:
                    top2_val = torch.topk(sim, k=2, dim=-1)[0]
                    margin = (top2_val[..., 0] - top2_val[..., 1]).mean().item()
                    diag[f"vayusphere_{prefix}_assignment_margin"] = margin

                # Per-token entropy
                per_token_ent = - (weights * torch.log(weights + 1e-9)).sum(dim=-1).mean().item()
                diag[f"vayusphere_{prefix}_per_token_entropy"] = per_token_ent

                # Global usage entropy
                soft_usage = weights.mean(dim=(0, 1, 2))
                usage_entropy = - (soft_usage * torch.log(soft_usage + 1e-9)).sum().item()
                diag[f"vayusphere_{prefix}_centroid_usage_entropy"] = usage_entropy

                # Top centroid ratio
                top_indices = weights.argmax(dim=-1)
                counts = torch.bincount(top_indices.view(-1), minlength=self.config.vayusphere_num_centroids)
                top_ratio = (counts.max().float() / top_indices.numel()).item()
                diag[f"vayusphere_{prefix}_top_centroid_usage_ratio"] = top_ratio

        return x_out, diag

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

        if not self.config.use_vayusphere:
            return q, k, diagnostics

        # If alpha and scale_alpha are both 0, return unchanged but can still log diagnostics if requested
        if self.alpha == 0 and self.scale_alpha == 0 and not return_diagnostics:
             return q, k, diagnostics

        # Check step filter for diagnostics
        collect_diag = return_diagnostics
        if collect_diag and self.config.vayusphere_diagnostics_every_n_steps > 0:
            if self.current_step % self.config.vayusphere_diagnostics_every_n_steps != 0:
                collect_diag = False

        if collect_diag:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()

        c_norm = F.normalize(self.centroids, dim=-1, eps=1e-6) if self.normalize else self.centroids

        if "q" in self.target:
            q, q_diag = self._apply_gate(q, c_norm, return_diagnostics=collect_diag, prefix="q")
            diagnostics.update(q_diag)

        if "k" in self.target:
            k, k_diag = self._apply_gate(k, c_norm, return_diagnostics=collect_diag, prefix="k")
            diagnostics.update(k_diag)

        if collect_diag:
            c_norms = torch.norm(self.centroids, dim=-1)
            diagnostics["vayusphere_centroid_norm_mean"] = c_norms.mean().item()
            diagnostics["vayusphere_centroid_norm_std"] = c_norms.std().item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time = time.time() - t0
            diagnostics["vayusphere_forward_time_ms"] = forward_time * 1000.0

        return q, k, diagnostics
