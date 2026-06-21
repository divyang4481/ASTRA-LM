import pytest
import torch
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM

def test_gradient_causality():
    config = ModelConfig(
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
        tie_word_embeddings=False, # Set to False to isolate input embedding gradients
        attention_type="chakra",
        local_window=16,
        sphere_buckets=8,
        nearby_buckets=1
    )
    
    model = DecoderForCausalLM(config)
    
    # We want to check gradients on the input embeddings
    # To do this, we can pass inputs, get the hidden states, and check gradients
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # Enable gradient tracking on embeddings
    model.zero_grad()
    
    # We'll hook into the embeddings to check gradients
    embeddings = model.model.embeddings.embedding
    
    # Let's perform a forward pass but manually run embedding lookup to track gradients
    x_emb = embeddings(input_ids).clone().detach().requires_grad_(True)
    
    # Now run the rest of the decoder model
    cos, sin = model.model.rope(10)
    hidden_states = x_emb
    
    for block in model.model.blocks:
        hidden_states, _ = block(hidden_states, cos, sin)
        
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    
    # Target position t
    t = 4
    loss = logits[0, t, :].sum()
    loss.backward()
    
    # The gradient of x_emb at positions > t must be exactly zero!
    grads = x_emb.grad[0] # Shape: [seq_len, d_model]
    
    for pos in range(10):
        grad_norm = grads[pos].norm().item()
        if pos <= t:
            # Past and current positions can have non-zero gradients
            pass
        else:
            # Future positions must have zero gradients
            assert grad_norm == 0.0, f"Future position {pos} has non-zero gradient {grad_norm} when loss is at position {t}"
