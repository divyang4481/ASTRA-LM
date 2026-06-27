import torch
import pytest
import numpy as np
from astra_lm.model.config import ModelConfig
from astra_lm.model.vayusphere_block_attention import VayuSphereBlockAttention

try:
    import triton
    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False

@pytest.mark.skipif(not HAS_TRITON, reason="Triton or CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [128, 100, 32]) # Divisible, non-divisible, smaller than block
@pytest.mark.parametrize("scorer", ["cosine", "linear"])
def test_triton_vs_pytorch_parity(batch_size, seq_len, scorer):
    block_size = 64
    d_model = 128
    n_heads = 4

    config = ModelConfig(
        vocab_size=1000, max_seq_len=256, d_model=d_model, n_layers=1, n_heads=n_heads, n_kv_heads=n_heads,
        attention_impl="vayusphere_block", vayu_block_size=block_size, vayu_top_m_blocks=2,
        vayu_pair_scorer=scorer
    )

    device = "cuda"
    # We test in half precision as that is the common production target for Triton kernels
    attn = VayuSphereBlockAttention(config).to(device).half()

    x = torch.randn(batch_size, seq_len, d_model, device=device).half()
    # Padded seq_len for cos/sin if needed internally
    total_seq_len = max(seq_len, block_size)
    cos = torch.randn(1, 1, total_seq_len, config.head_dim, device=device).half()
    sin = torch.randn(1, 1, total_seq_len, config.head_dim, device=device).half()

    # PyTorch reference
    with torch.no_grad():
        out_py, _ = attn(x, cos[:, :, :seq_len, :], sin[:, :, :seq_len, :], use_triton=False)

        # Triton
        out_triton, _ = attn(x, cos[:, :, :seq_len, :], sin[:, :, :seq_len, :], use_triton=True)

    # If seq_len not divisible by block_size, attn falls back to PyTorch internally and logs warning.
    # In that case out_py and out_triton should be identical.

    max_err = (out_py - out_triton).abs().max().item()
    mean_err = (out_py - out_triton).abs().mean().item()

    # Numerical tolerance for fp16
    # Triton kernel uses online softmax which may have slight diffs from offline PyTorch softmax
    assert max_err < 5e-2, f"Max error {max_err} too high for {scorer} scorer, seq_len {seq_len}"
    assert mean_err < 2e-3, f"Mean error {mean_err} too high for {scorer} scorer, seq_len {seq_len}"

@pytest.mark.skipif(not HAS_TRITON, reason="Triton or CUDA not available")
def test_triton_unsupported_scorer_fallback():
    config = ModelConfig(
        vocab_size=1000, max_seq_len=128, d_model=128, n_layers=1, n_heads=4, n_kv_heads=4,
        vayu_pair_scorer="mlp", vayu_use_triton_eval=True
    )
    device = "cuda"
    attn = VayuSphereBlockAttention(config).to(device)

    x = torch.randn(1, 128, 128, device=device)
    cos = torch.randn(1, 1, 128, config.head_dim, device=device)
    sin = torch.randn(1, 1, 128, config.head_dim, device=device)

    # Should not crash, and should fall back to PyTorch
    with torch.no_grad():
        out, _ = attn(x, cos, sin, use_triton=True)
    assert out.shape == (1, 128, 128)
