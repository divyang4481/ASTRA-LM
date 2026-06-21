import torch
import torch.nn as nn
from .config import ModelConfig

class FockFFN(nn.Module):
    """
    Structured basis-function FeedForward replacement.
    Approximates nonlinear activations via Chebyshev polynomial expansion.
    """
    def __init__(self, config: ModelConfig, degree: int = 4):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = int(config.mlp_ratio * config.d_model)
        self.degree = degree
        
        self.in_proj = nn.Linear(self.d_model, self.d_ff, bias=config.bias)
        self.out_proj = nn.Linear(self.d_ff, self.d_model, bias=config.bias)
        self.residual_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        
        # Chebyshev expansion coefficients for each FFN channel
        self.coeff = nn.Parameter(torch.randn(self.d_ff, degree + 1) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        # Project and map to [-1, 1] using tanh to prevent divergence of Chebyshev recurrence
        u = torch.tanh(self.in_proj(x)) # [batch, seq_len, d_ff]
        
        # Compute Chebyshev polynomials recursively
        # T_0(u) = 1
        # T_1(u) = u
        # T_{k+1}(u) = 2*u*T_k(u) - T_{k-1}(u)
        t_polys = []
        t_polys.append(torch.ones_like(u)) # T_0
        if self.degree >= 1:
            t_polys.append(u) # T_1
        
        for k in range(2, self.degree + 1):
            t_next = 2 * u * t_polys[k - 1] - t_polys[k - 2]
            t_polys.append(t_next)
            
        # Stack polynomials: [batch, seq_len, d_ff, degree + 1]
        t_stacked = torch.stack(t_polys, dim=-1)
        
        # Apply learnable coefficients and sum:
        # self.coeff has shape [d_ff, degree + 1]
        # output = sum_k (coeff_k * T_k)
        coeff_expanded = self.coeff.unsqueeze(0).unsqueeze(0) # [1, 1, d_ff, degree + 1]
        ffn_hidden = (t_stacked * coeff_expanded).sum(dim=-1) # [batch, seq_len, d_ff]
        
        # Project back and add residual linear path
        return self.out_proj(ffn_hidden) + self.residual_proj(x)
