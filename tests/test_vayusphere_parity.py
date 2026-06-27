import pytest
import torch
import torch.nn.functional as F
from astra_lm.model.config import ModelConfig
from astra_lm.model.vayusphere_block_attention import VayuSphereBlockAttention

def get_test_config(attention_impl="vayusphere_block", vayu_block_size=16, vayu_top_m_blocks=2):
    return ModelConfig(
        vocab_size=100,
        max_seq_len=1024,
        n_layers=1,
        d_model=32,
        n_heads=2,
        n_kv_heads=2,
        vayu_block_size=vayu_block_size,
        vayu_top_m_blocks=vayu_top_m_blocks,
        attention_impl=attention_impl,
        vayu_pair_scorer="linear",
        vayu_force_local_blocks=True,
        vayu_route_policy="current_prev_semantic"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
@pytest.mark.parametrize("seq_len", [32, 48, 64]) # 16 is block_size, test divisible and non-divisible
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("scorer", ["cosine", "linear"])
def test_vayusphere_triton_parity(seq_len, dtype, scorer):
    device = "cuda"
    batch_size = 2
    block_size = 16
    top_m = 2

    config = get_test_config(vayu_block_size=block_size, vayu_top_m_blocks=top_m)
    config.vayu_pair_scorer = scorer

    model = VayuSphereBlockAttention(config).to(device).to(dtype).eval()

    # Inputs
    hidden_states = torch.randn(batch_size, seq_len, config.d_model, device=device, dtype=dtype)

    # RoPE components (dummy)
    cos = torch.ones(1, 1, 1024, config.head_dim, device=device, dtype=dtype)
    sin = torch.zeros(1, 1, 1024, config.head_dim, device=device, dtype=dtype)

    # 1. PyTorch Reference path
    with torch.no_grad():
        out_ref, _ = model(hidden_states, cos, sin, use_triton=False)

    # 2. Triton path
    with torch.no_grad():
        out_triton, _ = model(hidden_states, cos, sin, use_triton=True)

    # Assertions
    # Tolerance depends on dtype
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    rtol = 1e-3 if dtype == torch.float32 else 1e-2

    torch.testing.assert_close(out_triton, out_ref, atol=atol, rtol=rtol)

def test_causality_and_padding():
    device = "cpu"
    seq_len = 32
    block_size = 16
    top_m = 2
    config = get_test_config(vayu_block_size=block_size, vayu_top_m_blocks=top_m)

    model = VayuSphereBlockAttention(config).to(device).eval()

    hidden_states = torch.randn(1, seq_len, config.d_model, device=device)
    hidden_states.requires_grad = True

    cos = torch.ones(1, 1, 1024, config.head_dim, device=device)
    sin = torch.zeros(1, 1, 1024, config.head_dim, device=device)

    out, _ = model(hidden_states, cos, sin, use_triton=False)

    # Check no-future leakage: gradient of out[0, i, :] w.r.t hidden_states[0, j, :] should be 0 if j > i
    for i in [0, 15, 16, 31]:
        loss = out[0, i, :].sum()
        loss.backward(retain_graph=True)
        grad = hidden_states.grad[0]

        # Future gradients must be zero
        future_grad = grad[i+1:, :]
        assert torch.all(future_grad == 0), f"Leakage detected at step {i}: non-zero gradient from future tokens"

        hidden_states.grad.zero_()

def test_padding_masking():
    device = "cpu"
    block_size = 16
    config = get_test_config(vayu_block_size=block_size)
    model = VayuSphereBlockAttention(config).to(device).eval()

    # Sequence length 20, padded to 32 (2 blocks of 16)
    seq_len = 20
    hidden_states = torch.randn(1, seq_len, config.d_model, device=device)

    cos = torch.ones(1, 1, 1024, config.head_dim, device=device)
    sin = torch.zeros(1, 1, 1024, config.head_dim, device=device)

    out, _ = model(hidden_states, cos, sin, use_triton=False)

    assert out.shape == (1, seq_len, config.d_model)
    assert not torch.isnan(out).any()

def test_backward_gradients():
    device = "cpu"
    config = get_test_config()
    model = VayuSphereBlockAttention(config).to(device).train()

    seq_len = 32
    hidden_states = torch.randn(1, seq_len, config.d_model, device=device, requires_grad=True)
    cos = torch.ones(1, 1, 1024, config.head_dim, device=device)
    sin = torch.zeros(1, 1, 1024, config.head_dim, device=device)

    out, _ = model(hidden_states, cos, sin, use_triton=False)
    loss = out.pow(2).mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()

    # Check projection weights
    assert model.q_proj.weight.grad is not None
    assert model.k_proj.weight.grad is not None
    assert model.v_proj.weight.grad is not None

    # Check scorer weights
    assert model.scorer.w_cos.grad is not None
    assert model.scorer.w_dist.grad is not None
