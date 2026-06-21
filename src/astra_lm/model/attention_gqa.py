import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import ModelConfig
from .rope import apply_rotary_pos_emb

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = config.n_kv_groups
        self.max_seq_len = config.max_seq_len

        # Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # Return (out, q, k, v)
        batch_size, seq_len, _ = hidden_states.shape

        if seq_len > self.max_seq_len:
             raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Store for return before expansion
        q_out, k_out, v_out = q, k, v

        # Grouped-Query Attention: Expand K and V to match Q's heads
        k, v = self.expand_kv(k, v)

        # Manual Scaled Dot-Product Attention
        causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device).tril()
        if attention_mask is not None:
            # Combine causal mask with padding mask if provided
            causal_mask = causal_mask & (attention_mask > -1).squeeze(1)

        attn_output = self.scaled_dot_product_attention(
            q, k, v, 
            mask=causal_mask
        )

        return attn_output, q_out, k_out, v_out

    def expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand K/V heads to match Q heads for GQA calculation.
        """
        batch_size, n_kv_heads, seq_len, head_dim = k.shape
        if n_kv_heads == self.n_heads:
            return k, v
            
        k = k.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
        v = v.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
        
        k = k.reshape(batch_size, self.n_heads, seq_len, head_dim)
        v = v.reshape(batch_size, self.n_heads, seq_len, head_dim)
        return k, v

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard Matmul Attention.
        """
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
             if mask.dim() == 2:
                 mask = mask.unsqueeze(0).unsqueeze(1)
             attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        attn_weights_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights_probs = self.attn_dropout(attn_weights_probs)

        return torch.matmul(attn_weights_probs, v)
