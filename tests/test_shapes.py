import pytest
import torch
from astra_lm.model.config import ModelConfig
from astra_lm.model.embeddings import TokenEmbedding
from astra_lm.model.rope import RotaryEmbedding
from astra_lm.model.attention_gqa import GroupedQueryAttention
from astra_lm.model.chakra_attention import ChakraAttention
from astra_lm.model.decoder import DecoderForCausalLM

@pytest.fixture
def tiny_config():
    return ModelConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
        rope_base=10000.0,
        norm_eps=1e-5,
        bias=False,
        tie_word_embeddings=True,
        attention_type="chakra",
        local_window=16,
        sphere_buckets=8,
        nearby_buckets=1
    )

def test_embedding_shapes(tiny_config):
    emb = TokenEmbedding(tiny_config.vocab_size, tiny_config.d_model)
    x = torch.randint(0, tiny_config.vocab_size, (2, 10))
    out = emb(x)
    assert out.shape == (2, 10, tiny_config.d_model)

def test_rope_shapes(tiny_config):
    rope = RotaryEmbedding(tiny_config.head_dim, tiny_config.max_seq_len)
    cos, sin = rope(10)
    assert cos.shape == (1, 1, 10, tiny_config.head_dim)
    assert sin.shape == (1, 1, 10, tiny_config.head_dim)

def test_gqa_shapes(tiny_config):
    gqa = GroupedQueryAttention(tiny_config)
    x = torch.randn(2, 10, tiny_config.d_model)
    rope = RotaryEmbedding(tiny_config.head_dim, tiny_config.max_seq_len)
    cos, sin = rope(10)
    out, q, k, v = gqa(x, cos, sin)
    assert out.shape == (2, tiny_config.n_heads, 10, tiny_config.head_dim)
    assert q.shape == (2, tiny_config.n_heads, 10, tiny_config.head_dim)
    assert k.shape == (2, tiny_config.n_kv_heads, 10, tiny_config.head_dim)
    assert v.shape == (2, tiny_config.n_kv_heads, 10, tiny_config.head_dim)

def test_chakra_shapes(tiny_config):
    chakra = ChakraAttention(tiny_config)
    x = torch.randn(2, 10, tiny_config.d_model)
    rope = RotaryEmbedding(tiny_config.head_dim, tiny_config.max_seq_len)
    cos, sin = rope(10)
    out, q, k, v, diags = chakra(x, cos, sin, return_diagnostics=True)
    assert out.shape == (2, tiny_config.n_heads, 10, tiny_config.head_dim)
    assert q.shape == (2, tiny_config.n_heads, 10, tiny_config.head_dim)
    assert k.shape == (2, tiny_config.n_kv_heads, 10, tiny_config.head_dim)
    assert v.shape == (2, tiny_config.n_kv_heads, 10, tiny_config.head_dim)
    assert "candidate_ratio" in diags
    assert "bucket_histogram" in diags

def test_decoder_causal_lm_shapes(tiny_config):
    model = DecoderForCausalLM(tiny_config)
    x = torch.randint(0, tiny_config.vocab_size, (2, 10))
    outputs = model(x, labels=x)
    assert outputs["logits"].shape == (2, 10, tiny_config.vocab_size)
    assert outputs["loss"] is not None
    assert isinstance(outputs["loss"].item(), float)
