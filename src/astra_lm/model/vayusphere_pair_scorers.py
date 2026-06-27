import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CosineOnlyScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # No extra parameters for cosine only

    def forward(self, cosine, log_dist, q_norm_mag, k_norm_mag):
        return torch.zeros_like(cosine)

class LinearPairScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_cos = nn.Parameter(torch.randn(1) * 0.02)
        self.w_dist = nn.Parameter(torch.randn(1) * 0.02)
        self.w_qnorm = nn.Parameter(torch.randn(1) * 0.02)
        self.w_knorm = nn.Parameter(torch.randn(1) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1))

        # Initialize distance weight to small negative value as suggested
        nn.init.constant_(self.w_dist, -0.01)

    def forward(self, cosine, log_dist, q_norm_mag, k_norm_mag):
        # cosine: [..., L_q, L_k]
        # log_dist: [..., L_q, L_k]
        # q_norm_mag: [..., L_q, 1]
        # k_norm_mag: [..., 1, L_k]

        score_delta = (
            self.w_cos * cosine
            + self.w_dist * log_dist
            + self.w_qnorm * q_norm_mag
            + self.w_knorm * k_norm_mag
            + self.bias
        )
        return score_delta

class TinyMLPScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = getattr(config, "vayu_mlp_hidden", 16)
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        # Initialize weights small
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cosine, log_dist, q_norm_mag, k_norm_mag):
        # [..., Lq, Lk, 4]
        # This might be memory intensive for large Lq, Lk
        # Need to be careful. In block attention Lq=64, Lk=64*top_m

        # Broadcast features
        q_norm_mag = q_norm_mag.expand_as(cosine)
        k_norm_mag = k_norm_mag.expand_as(cosine)

        features = torch.stack([cosine, log_dist, q_norm_mag, k_norm_mag], dim=-1)
        return self.net(features).squeeze(-1)

class RBFKANScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_centers = getattr(config, "vayu_rbf_centers", 8)
        self.gamma = getattr(config, "vayu_rbf_gamma", 8.0)

        # RBF centers in [0, 1] or similar range
        self.centers = nn.Parameter(torch.linspace(-1, 1, self.num_centers))
        # 4 input features, each has num_centers RBFs, then linear combination
        self.weights = nn.Parameter(torch.randn(4, self.num_centers) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, cosine, log_dist, q_norm_mag, k_norm_mag):
        # Broadcast features
        q_norm_mag = q_norm_mag.expand_as(cosine)
        k_norm_mag = k_norm_mag.expand_as(cosine)

        features = torch.stack([cosine, log_dist, q_norm_mag, k_norm_mag], dim=-1) # [..., 4]

        # [..., 4, 1] - [num_centers] -> [..., 4, num_centers]
        diff = features.unsqueeze(-1) - self.centers
        rbf_out = torch.exp(-self.gamma * diff.pow(2))

        # [..., 4, num_centers] * [4, num_centers] -> sum over 4, num_centers
        score_delta = (rbf_out * self.weights).sum(dim=(-1, -2)) + self.bias
        return score_delta

def get_vayu_pair_scorer(config):
    scorer_type = getattr(config, "vayu_pair_scorer", "linear")
    if scorer_type == "cosine":
        return CosineOnlyScorer(config)
    elif scorer_type == "linear":
        return LinearPairScorer(config)
    elif scorer_type == "mlp":
        return TinyMLPScorer(config)
    elif scorer_type == "rbfkan":
        return RBFKANScorer(config)
    else:
        raise ValueError(f"Unknown vayu_pair_scorer: {scorer_type}")
