import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .config import ModelConfig
from .rope import apply_rotary_pos_emb
from .sphere_bucket import SphereBucketer

class ChakraAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = config.n_kv_groups
        self.max_seq_len = config.max_seq_len
        
        self.local_window = config.local_window
        self.nearby_buckets = config.nearby_buckets

        # Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Spherical bucketing module
        self.bucketer = SphereBucketer(config.sphere_buckets, config.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]: 
        """
        Returns:
            attn_output: [batch, n_heads, seq_len, head_dim]
            q_out, k_out, v_out: raw Q, K, V (before head expansion)
            diagnostics: routing information and candidate masks
        """
        batch_size, seq_len, _ = hidden_states.shape

        if seq_len > self.max_seq_len:
             raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # 1. Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Save before expansion for return
        q_out, k_out, v_out = q, k, v

        # 4. GQA Expansion
        k_expanded, v_expanded = self.expand_kv(k, v)

        # 5. CHAKRA Candidate Routing
        candidate_mask, diagnostics = self.bucketer(
            q=q, 
            k=k_expanded, 
            local_window=self.local_window, 
            nearby_buckets=self.nearby_buckets,
            return_diagnostics=return_diagnostics
        )
        
        # If extra attention mask is passed (like padding mask), combine it
        if attention_mask is not None:
            # Combine
            if attention_mask.dim() == 2:
                attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2) > -1
            else:
                attention_mask_expanded = attention_mask > -1
            candidate_mask = candidate_mask & attention_mask_expanded

        # 6. Scaled Dot Product Attention on Candidates
        attn_output = self.scaled_dot_product_attention_masked(
            q, k_expanded, v_expanded, mask=candidate_mask
        )

        return attn_output, q_out, k_out, v_out, diagnostics

    def expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_kv_heads, seq_len, head_dim = k.shape
        if n_kv_heads == self.n_heads:
            return k, v
            
        k = k.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
        v = v.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
        
        k = k.reshape(batch_size, self.n_heads, seq_len, head_dim)
        v = v.reshape(batch_size, self.n_heads, seq_len, head_dim)
        return k, v

    def scaled_dot_product_attention_masked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        # Q @ K^T / sqrt(d)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Mask out non-candidate logits using masked_fill
        # mask shape: [batch, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        # Softmax and Dropout
        attn_weights_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights_probs = self.attn_dropout(attn_weights_probs)

        # Matmul with V
        return torch.matmul(attn_weights_probs, v)
