import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _vayusphere_block_attn_fwd_kernel(
        Q_NORM, K_NORM, V,
        ROUTE_IDX,
        Q_MAG, K_MAG,
        W_COS, W_DIST, W_QNORM, W_KNORM, BIAS,
        OUT,
        stride_qh, stride_qt, stride_qd,
        stride_kh, stride_kt, stride_kd,
        stride_vh, stride_vt, stride_vd,
        stride_rh, stride_rb, stride_rm,
        stride_oh, stride_ot, stride_od,
        n_heads, seq_len, head_dim,
        block_size: tl.constexpr,
        top_m: tl.constexpr,
        temperature,
        delta_scale,
        HAS_LINEAR: tl.constexpr,
    ):
        # Program ID
        batch_pid = tl.program_id(0)
        head_pid = tl.program_id(1)
        block_pid = tl.program_id(2)

        # Offsets
        offs_m = block_pid * block_size + tl.arange(0, block_size)
        offs_n = tl.arange(0, block_size)
        offs_d = tl.arange(0, head_dim)

        # Load Q block [block_size, head_dim]
        q_norm_ptr = Q_NORM + batch_pid * (n_heads * seq_len * head_dim) + head_pid * (seq_len * head_dim) + (block_pid * block_size * head_dim) + offs_m[:, None] * head_dim + offs_d[None, :]
        q_norm = tl.load(q_norm_ptr)

        q_mag = 0.0
        if HAS_LINEAR:
            q_mag_ptr = Q_MAG + batch_pid * (n_heads * seq_len) + head_pid * seq_len + block_pid * block_size + tl.arange(0, block_size)
            q_mag = tl.load(q_mag_ptr) # [block_size]

        # Online softmax state
        m_i = tl.full([block_size], -float('inf'), dtype=tl.float32)
        l_i = tl.zeros([block_size], dtype=tl.float32)
        acc = tl.zeros([block_size, head_dim], dtype=tl.float32)

        # Iterate over routed key blocks
        for m in range(top_m):
            # Load route index
            route_idx_ptr = ROUTE_IDX + batch_pid * (n_heads * (seq_len // block_size) * top_m) + head_pid * ((seq_len // block_size) * top_m) + block_pid * top_m + m
            k_block_idx = tl.load(route_idx_ptr)

            # Load K block
            k_norm_ptr = K_NORM + batch_pid * (n_heads * seq_len * head_dim) + head_pid * (seq_len * head_dim) + (k_block_idx * block_size * head_dim) + offs_n[None, :] * head_dim + offs_d[:, None]
            k_norm = tl.load(k_norm_ptr) # [head_dim, block_size]

            # Compute QK^T (cosine similarity)
            qk = tl.dot(q_norm, k_norm) # [block_size, block_size]

            # Token positions for causal masking and distance
            q_pos = block_pid * block_size + tl.arange(0, block_size)
            k_pos = k_block_idx * block_size + tl.arange(0, block_size)

            # Causal mask
            mask = q_pos[:, None] >= k_pos[None, :]

            # Scorers
            logits = qk / temperature

            if HAS_LINEAR:
                # k_mag
                k_mag_ptr = K_MAG + batch_pid * (n_heads * seq_len) + head_pid * seq_len + k_block_idx * block_size + tl.arange(0, block_size)
                k_mag = tl.load(k_mag_ptr) # [block_size]

                # log_dist
                # distance = max(q_pos - k_pos, 0)
                # log_dist = log1p(distance) / log1p(seq_len)
                dist = q_pos[:, None] - k_pos[None, :]
                dist = tl.where(dist < 0, 0, dist)
                log_dist = tl.log(1.0 + dist.to(tl.float32)) / tl.log(1.0 + seq_len)

                score_delta = (
                    W_COS * qk +
                    W_DIST * log_dist +
                    W_QNORM * q_mag[:, None] +
                    W_KNORM * k_mag[None, :] +
                    BIAS
                )
                logits += delta_scale * score_delta

            logits = tl.where(mask, logits, -float('inf'))

            # Online softmax
            m_ij = tl.max(logits, 1)
            p = tl.exp(logits - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)

            l_i = l_i * alpha + l_ij * beta

            # Load V block
            v_ptr = V + batch_pid * (n_heads * seq_len * head_dim) + head_pid * (seq_len * head_dim) + (k_block_idx * block_size * head_dim) + offs_n[:, None] * head_dim + offs_d[None, :]
            v = tl.load(v_ptr) # [block_size, head_dim]

            # Update accumulator
            # p is [block_size, block_size], v is [block_size, head_dim]
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v) * beta[:, None]
            m_i = m_i_new

        # Finalize and store
        acc = acc / l_i[:, None]
        out_ptr = OUT + batch_pid * (n_heads * seq_len * head_dim) + head_pid * (seq_len * head_dim) + (block_pid * block_size * head_dim) + offs_m[:, None] * head_dim + offs_d[None, :]
        tl.store(out_ptr, acc.to(OUT.dtype.element_ty))

def triton_vayusphere_block_attention_forward(
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    v: torch.Tensor,
    route_idx: torch.Tensor,
    q_norm_mag: Optional[torch.Tensor] = None,
    k_norm_mag: Optional[torch.Tensor] = None,
    linear_weights: Optional[nn.Module] = None,
    temperature: float = 1.0,
    delta_scale: float = 0.0,
    block_size: int = 64,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise ImportError("Triton not installed.")

    batch_size, n_heads, seq_len, head_dim = q_norm.shape
    num_blocks = seq_len // block_size
    top_m = route_idx.shape[-1]

    out = torch.empty_like(v)

    has_linear = linear_weights is not None and hasattr(linear_weights, 'w_cos')

    w_cos, w_dist, w_qnorm, w_knorm, bias = 0.0, 0.0, 0.0, 0.0, 0.0
    if has_linear:
        w_cos = linear_weights.w_cos.item()
        w_dist = linear_weights.w_dist.item()
        w_qnorm = linear_weights.w_qnorm.item()
        w_knorm = linear_weights.w_knorm.item()
        bias = linear_weights.bias.item()

    grid = (batch_size, n_heads, num_blocks)

    # Q_MAG, K_MAG expect [batch, heads, seq_len]
    q_mag_input = q_norm_mag.squeeze(-1) if q_norm_mag is not None else None
    k_mag_input = k_norm_mag.squeeze(-1) if k_norm_mag is not None else None

    _vayusphere_block_attn_fwd_kernel[grid](
        q_norm, k_norm, v,
        route_idx,
        q_mag_input, k_mag_input,
        w_cos, w_dist, w_qnorm, w_knorm, bias,
        out,
        q_norm.stride(1), q_norm.stride(2), q_norm.stride(3),
        k_norm.stride(1), k_norm.stride(2), k_norm.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        route_idx.stride(1), route_idx.stride(2), route_idx.stride(3),
        out.stride(1), out.stride(2), out.stride(3),
        n_heads, seq_len, head_dim,
        block_size=block_size,
        top_m=top_m,
        temperature=float(temperature),
        delta_scale=float(delta_scale),
        HAS_LINEAR=has_linear,
    )

    return out
