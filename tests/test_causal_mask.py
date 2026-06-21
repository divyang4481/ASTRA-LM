import pytest
import torch
from astra_lm.model.config import ModelConfig
from astra_lm.model.sphere_bucket import SphereBucketer

def test_causal_mask_structure():
    # Setup bucketer
    num_buckets = 8
    head_dim = 16
    local_window = 4
    nearby_buckets = 1
    
    bucketer = SphereBucketer(num_buckets, head_dim)
    
    batch = 1
    n_heads = 2
    seq_len = 10
    
    q = torch.randn(batch, n_heads, seq_len, head_dim)
    k = torch.randn(batch, n_heads, seq_len, head_dim)
    
    candidate_mask, diags = bucketer(q, k, local_window, nearby_buckets)
    
    # Assert that all elements above the main diagonal are False (causal constraint)
    for b in range(batch):
        for h in range(n_heads):
            mask = candidate_mask[b, h]
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert mask[i, j].item() is False, f"Non-causal connection found: token {i} attends to future token {j}"
