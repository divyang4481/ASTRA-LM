import torch
from astra_lm.model.config import ModelConfig
from astra_lm.model.attention_sdpa import SDPAGPTAttention

def test_sdpa_attention_shapes():
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=128,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2
    )
    attn = SDPAGPTAttention(config)

    x = torch.randn(2, 64, 128)
    cos = torch.randn(1, 1, 64, 32)
    sin = torch.randn(1, 1, 64, 32)

    out, q, k, v = attn(x, cos, sin)

    assert out.shape == (2, 4, 64, 32)
    assert q.shape == (2, 4, 64, 32)
    assert k.shape == (2, 2, 64, 32)
    assert v.shape == (2, 2, 64, 32)

def test_sdpa_causality():
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=128,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4
    )
    attn = SDPAGPTAttention(config).eval()

    x = torch.randn(1, 10, 128)
    cos = torch.randn(1, 1, 10, 32)
    sin = torch.randn(1, 1, 10, 32)

    # Run once
    out1, _, _, _ = attn(x, cos, sin)

    # Change last token
    x2 = x.clone()
    x2[0, -1, :] += 1.0
    out2, _, _, _ = attn(x2, cos, sin)

    # First 9 tokens should be identical
    assert torch.allclose(out1[:, :, :-1, :], out2[:, :, :-1, :], atol=1e-5)
    # Last token should be different
    assert not torch.allclose(out1[:, :, -1, :], out2[:, :, -1, :], atol=1e-5)

if __name__ == "__main__":
    test_sdpa_attention_shapes()
    test_sdpa_causality()
    print("SDPA Attention tests passed!")
