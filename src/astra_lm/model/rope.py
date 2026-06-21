import torch
import torch.nn as nn
from typing import Tuple

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")

        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute positions up to max_seq_len
        # shape of t: [max_seq_len]
        t = torch.arange(max_seq_len).float()

        # freqs shape: [max_seq_len, head_dim // 2]
        freqs = torch.outer(t, self.inv_freq)

        # Expand frequencies for both sin and cos
        # freq_emb shape: [max_seq_len, head_dim]
        freqs_emb = torch.cat((freqs, freqs), dim=-1)

        # We need sin and cos to be easily broadcastable to [batch, n_heads, seq_len, head_dim]
        # Currently they are [max_seq_len, head_dim]
        self.register_buffer("cos_cached", freqs_emb.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Return sliced precomputed values, reshaped for broadcasting
        # [1, 1, seq_len, head_dim]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        return cos, sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to query and key tensors.
    """
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
