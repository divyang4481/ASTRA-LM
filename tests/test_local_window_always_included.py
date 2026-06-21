import pytest
import torch
from astra_lm.model.sphere_bucket import SphereBucketer

def test_local_window_preservation():
    num_buckets = 16
    head_dim = 8
    local_window = 12
    nearby_buckets = 0 # No neighbors, strict bucketing
    
    bucketer = SphereBucketer(num_buckets, head_dim)
    
    # Generate random queries and keys where bucketing is unlikely to match
    # but local window must be preserved
    batch = 2
    n_heads = 4
    seq_len = 32
    
    q = torch.randn(batch, n_heads, seq_len, head_dim)
    k = torch.randn(batch, n_heads, seq_len, head_dim)
    
    candidate_mask, diags = bucketer(q, k, local_window, nearby_buckets)
    
    # Assert coverage is 1.0
    assert diags["local_window_coverage"] == 1.0
    
    # Manually check every causal and local window cell is True in candidate_mask
    causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool).tril()
    window_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=-local_window + 1)
    local_expected = causal_mask & window_mask
    
    for b in range(batch):
        for h in range(n_heads):
            mask = candidate_mask[b, h]
            # Wherever local_expected is True, mask MUST be True
            assert torch.all(mask[local_expected] == True)
