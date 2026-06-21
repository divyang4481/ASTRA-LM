import pytest
import torch
from astra_lm.model.sphere_bucket import SphereBucketer

def test_candidate_mask_routing_logic():
    num_buckets = 8
    head_dim = 4
    local_window = 2
    nearby_buckets = 1
    
    bucketer = SphereBucketer(num_buckets, head_dim)
    
    # Manually fix centroids to orthogonal vectors to make bucket assignment deterministic
    # We will initialize centroids as eye-like matrix
    centroids = torch.zeros(num_buckets, head_dim)
    centroids[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    centroids[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    centroids[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    centroids[3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    # The rest are copies or simple vectors
    centroids[4] = torch.tensor([-1.0, 0.0, 0.0, 0.0])
    centroids[5] = torch.tensor([0.0, -1.0, 0.0, 0.0])
    centroids[6] = torch.tensor([0.0, 0.0, -1.0, 0.0])
    centroids[7] = torch.tensor([0.0, 0.0, 0.0, -1.0])
    
    bucketer.centroids.data.copy_(centroids)
    
    # Query: token 0 -> bucket 0, token 1 -> bucket 2
    q = torch.zeros(1, 1, 2, head_dim)
    q[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0]) # projects to bucket 0
    q[0, 0, 1] = torch.tensor([0.0, 0.0, 1.0, 0.0]) # projects to bucket 2
    
    # Key: token 0 -> bucket 0, token 1 -> bucket 1
    k = torch.zeros(1, 1, 2, head_dim)
    k[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0]) # projects to bucket 0
    k[0, 0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0]) # projects to bucket 1
    
    # Run bucketer
    candidate_mask, diags = bucketer(q, k, local_window, nearby_buckets, return_diagnostics=True)
    
    # Let's inspect q_buckets and k_buckets
    q_b = diags["q_buckets"][0, 0]
    k_b = diags["k_buckets"][0, 0]
    
    assert q_b[0].item() == 0
    assert q_b[1].item() == 2
    assert k_b[0].item() == 0
    assert k_b[1].item() == 1
    
    # Let's check candidate mask connections
    # Causal layout: query 0 can only attend to key 0
    # Query 1 can attend to keys 0, 1
    # For query 1 (bucket 2): nearby_buckets = 1 means allowed buckets are {1, 2, 3}.
    # Key 0 is bucket 0 -> not in {1, 2, 3}
    # Key 1 is bucket 1 -> in {1, 2, 3} (matching neighbor!)
    # Local window is 2, meaning query 1 can attend to key 0 (local_window = 2 covers offset 1, which is key 0).
    # So:
    # (0, 0): query 0 (bucket 0) and key 0 (bucket 0) -> matches! Mask should be True.
    # (0, 1): query 0 and key 1 (future) -> False (causal constraint).
    # (1, 0): query 1 and key 0 -> local window cover -> True.
    # (1, 1): query 1 (bucket 2) and key 1 (bucket 1) -> neighbor match -> True.
    
    assert candidate_mask[0, 0, 0, 0].item() is True
    assert candidate_mask[0, 0, 0, 1].item() is False
    assert candidate_mask[0, 0, 1, 0].item() is True
    assert candidate_mask[0, 0, 1, 1].item() is True
