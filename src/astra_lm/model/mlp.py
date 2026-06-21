import torch
import torch.nn as nn
from .config import ModelConfig

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = int(config.mlp_ratio * config.d_model)
        self.bias = config.bias

        self.w1 = nn.Linear(self.d_model, self.d_ff, bias=self.bias)
        self.act = nn.SiLU() if config.activation == "silu" else nn.GELU()
        self.w2 = nn.Linear(self.d_ff, self.d_model, bias=self.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x
