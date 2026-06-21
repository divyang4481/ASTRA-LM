import torch
import torch.nn as nn
import math
from .config import ModelConfig

class IndraPhaseLayer(nn.Module):
    """
    Lightweight phase/magnitude gate on real tensors.
    Modulates hidden states by a magnitude scale and a phase angle rotation.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Projections to compute magnitude and phase angles
        self.mag_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.phase_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        # Magnitude gating (non-negative scale)
        mag = torch.sigmoid(self.mag_proj(x))
        
        # Phase gating (angles mapped to [-pi, pi])
        phase = torch.tanh(self.phase_proj(x)) * math.pi
        
        # Apply gating: Magnitude scaling * input * Cosine of Phase rotation
        return mag * x * torch.cos(phase)
