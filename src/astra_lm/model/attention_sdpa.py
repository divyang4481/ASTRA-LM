import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import ModelConfig
from .rope import apply_rotary_pos_emb

class SDPAGPTAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = config.n_heads // config.n_kv_heads
        self.max_seq_len = config.max_seq_len

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=config.bias)

        self.attn_dropout = config.attention_dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        if config.use_learned_attention_temp:
            self.log_attn_temp = nn.Parameter(torch.zeros(self.n_heads))
        else:
            self.register_parameter("log_attn_temp", None)

    def get_qkv(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = q.shape

        if self.log_attn_temp is not None:
            # log_attn_temp: [H] -> [1, H, 1, 1]
            scale = torch.exp(self.log_attn_temp).view(1, n_heads, 1, 1)
            q = q * scale

        k_sdpa, v_sdpa = self.expand_kv(k, v)

        if attention_mask is not None:
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device).tril()
            combined_mask = causal_mask & (attention_mask > -1)

            attn_output = F.scaled_dot_product_attention(
                q, k_sdpa, v_sdpa,
                attn_mask=combined_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k_sdpa, v_sdpa,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True
            )
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.get_qkv(hidden_states, cos, sin)
        q_out, k_out, v_out = q, k, v

        attn_output = self.compute_attention(q, k, v, attention_mask)

        # attn_output: [B, H, S, Hd]
        return attn_output, q_out, k_out, v_out

    def expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_kv_heads == self.n_heads:
            return k, v
        k = k.repeat_interleave(self.n_kv_groups, dim=1)
        v = v.repeat_interleave(self.n_kv_groups, dim=1)
        return k, v

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Standard SDPA for AKASHA
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False if mask is not None else False # AKASHA anchors usually don't use causal mask
        )
