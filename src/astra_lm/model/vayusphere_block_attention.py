import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import logging

from .config import ModelConfig
from .vayusphere_block_router import build_vayusphere_block_routes
from .vayusphere_pair_scorers import get_vayu_pair_scorer
from .rope import apply_rotary_pos_emb

logger = logging.getLogger(__name__)

class VayuSphereBlockAttention(nn.Module):
    """
    VayuSphere-Fused Block Attention v0.1.
    Experimental attention path with block-sparse routing and learned pair scoring.

    Known limitations v0.1:
    - Supports Multi-Head Attention (MHA) only (n_heads == n_kv_heads).
    - Triton fused path supports only 'cosine' and 'linear' scorers.
    - Triton fused path is implemented for forward pass only.
    - No optimized KV-cache support for autoregressive generation.
    - Internal padding used for sequence lengths not divisible by block_size.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        if self.n_heads != self.n_kv_heads:
            raise ValueError(
                "VayuSphereBlockAttention v0.1 supports MHA only (n_heads == n_kv_heads). "
                f"Found n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}. "
                "Set n_kv_heads equal to n_heads or use 'sdpa' implementation."
            )

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

        self.block_size = config.vayu_block_size
        self.top_m_blocks = config.vayu_top_m_blocks

        self.scorer = get_vayu_pair_scorer(config)

        self.delta_scale = nn.Parameter(torch.tensor(config.vayu_delta_scale_init))
        self.temperature = nn.Parameter(torch.tensor(config.vayu_temperature_init))

        self._triton_warned = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_triton: bool = False,
        return_stats: bool = False,
        past_key_value: Optional[Any] = None, # Not supported yet
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if past_key_value is not None:
             raise NotImplementedError("VayuSphereBlockAttention v0.1 does not support optimized KV-cache generation yet.")

        batch_size, seq_len, _ = hidden_states.shape

        # 1. Padding if seq_len not divisible by block_size
        pad_len = (self.block_size - seq_len % self.block_size) % self.block_size
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            # attention_mask handling for padding
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                     attention_mask = F.pad(attention_mask, (0, pad_len, 0, pad_len), value=-10000.0)
                elif attention_mask.dim() == 2:
                     attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

            curr_seq_len = seq_len + pad_len
        else:
            curr_seq_len = seq_len

        # Triton Fallback Check
        if use_triton:
            if self.training:
                # v0.1 Triton is forward-only
                use_triton = False
            elif not torch.cuda.is_available():
                use_triton = False
            elif self.config.vayu_pair_scorer not in ["cosine", "linear"]:
                if not self._triton_warned:
                    logger.warning(f"VayuSphere Triton v0.1 supports only cosine and linear scorers. Falling back to PyTorch path for scorer={self.config.vayu_pair_scorer}.")
                    self._triton_warned = True
                use_triton = False
            elif pad_len > 0:
                if not self._triton_warned:
                    logger.warning("VayuSphere Triton v0.1 fallback to PyTorch path for padded sequence (seq_len % block_size != 0).")
                    self._triton_warned = True
                use_triton = False
            elif curr_seq_len < self.block_size:
                use_triton = False

        # 2. QKV Projection
        q = self.q_proj(hidden_states).view(batch_size, curr_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, curr_seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, curr_seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE
        if pad_len > 0:
             if cos.size(2) < curr_seq_len:
                 # Normally rope frequencies are pre-calculated for max_seq_len
                 # but for safety in tests we pad them
                 cos = F.pad(cos, (0, 0, 0, curr_seq_len - cos.size(2)), value=1.0)
                 sin = F.pad(sin, (0, 0, 0, curr_seq_len - sin.size(2)), value=0.0)
             else:
                 cos = cos[:, :, :curr_seq_len, :]
                 sin = sin[:, :, :curr_seq_len, :]

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Routing
        route_idx, route_stats = build_vayusphere_block_routes(
            q, k,
            block_size=self.block_size,
            top_m_blocks=self.top_m_blocks,
            force_local_blocks=self.config.vayu_force_local_blocks,
            route_policy=self.config.vayu_route_policy
        )

        if use_triton:
            try:
                from .triton_vayusphere_block_attention import triton_vayusphere_block_attention_forward
                q_norm = F.normalize(q, p=2, dim=-1)
                k_norm = F.normalize(k, p=2, dim=-1)
                q_mag = torch.norm(q, p=2, dim=-1, keepdim=True)
                k_mag = torch.norm(k, p=2, dim=-1, keepdim=True)

                attn_out = triton_vayusphere_block_attention_forward(
                    q_norm=q_norm,
                    k_norm=k_norm,
                    v=v,
                    route_idx=route_idx,
                    q_norm_mag=q_mag,
                    k_norm_mag=k_mag,
                    linear_weights=self.scorer,
                    temperature=self.temperature,
                    delta_scale=self.delta_scale,
                    block_size=self.block_size
                )
                attn_out = attn_out.transpose(1, 2).reshape(batch_size, curr_seq_len, self.d_model)
            except (ImportError, RuntimeError) as e:
                if not self._triton_warned:
                    logger.warning(f"Triton failed or not installed, falling back to PyTorch: {e}")
                    self._triton_warned = True
                use_triton = False

        if not use_triton:
            attn_out = self.pytorch_forward(q, k, v, route_idx)
            attn_out = attn_out.transpose(1, 2).reshape(batch_size, curr_seq_len, self.d_model)

        # 5. Output Projection
        attn_out = self.o_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)

        # 6. Unpad
        if pad_len > 0:
            attn_out = attn_out[:, :seq_len, :]

        diagnostics = {}
        if return_stats or self.config.vayu_log_route_stats:
            diagnostics.update(route_stats)

        return attn_out, diagnostics

    def pytorch_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = q.shape
        num_blocks = seq_len // self.block_size

        # Reshape to blocks
        q_blocks = q.reshape(batch_size, n_heads, num_blocks, self.block_size, head_dim).unsqueeze(3)
        k_blocks_all = k.reshape(batch_size, n_heads, num_blocks, self.block_size, head_dim)
        v_blocks_all = v.reshape(batch_size, n_heads, num_blocks, self.block_size, head_dim)

        # Advanced indexing for block gather
        B, H, NB, BS, D = k_blocks_all.shape
        b_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
        h_idx = torch.arange(H, device=q.device).view(1, H, 1, 1)

        k_blocks = k_blocks_all[b_idx, h_idx, route_idx]
        v_blocks = v_blocks_all[b_idx, h_idx, route_idx]

        q_norm = F.normalize(q_blocks, p=2, dim=-1)
        k_norm = F.normalize(k_blocks, p=2, dim=-1)

        # Cosine similarity
        cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))

        # Learned scoring features
        q_mag = torch.norm(q_blocks, p=2, dim=-1, keepdim=True)
        k_mag = torch.norm(k_blocks, p=2, dim=-1, keepdim=True)

        # Relative distance
        q_pos = torch.arange(seq_len, device=q.device).view(num_blocks, self.block_size)
        k_pos_all = torch.arange(seq_len, device=q.device).view(num_blocks, self.block_size)
        k_pos = k_pos_all[route_idx].view(batch_size, n_heads, num_blocks, self.top_m_blocks, self.block_size)

        dist = q_pos.view(1, 1, num_blocks, 1, self.block_size, 1) - k_pos.unsqueeze(-2)
        dist = torch.clamp(dist, min=0)
        log_dist = torch.log1p(dist.to(torch.float32)) / math.log1p(seq_len)

        score_delta = self.scorer(
            cosine,
            log_dist,
            q_mag,
            k_mag.transpose(-1, -2)
        )

        logits = cosine / self.temperature + self.delta_scale * score_delta

        # Causal Masking
        causal_mask = q_pos.view(1, 1, num_blocks, 1, self.block_size, 1) >= k_pos.unsqueeze(-2)
        logits = logits.masked_fill(~causal_mask, -10000.0)

        # Softmax and Aggregation
        logits = logits.permute(0, 1, 2, 4, 3, 5).reshape(batch_size, n_heads, num_blocks, self.block_size, -1)
        attn_weights = F.softmax(logits, dim=-1)

        v_blocks = v_blocks.reshape(batch_size, n_heads, num_blocks, -1, head_dim)
        attn_out_blocks = torch.matmul(attn_weights, v_blocks)

        return attn_out_blocks.reshape(batch_size, n_heads, seq_len, head_dim)
