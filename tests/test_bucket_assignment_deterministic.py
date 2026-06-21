import pytest
import torch
from astra_lm.model.sphere_bucket import SphereBucketer

def test_bucket_projection_determinism():
    num_buckets = 16
    head_dim = 8
    
    bucketer = SphereBucketer(num_buckets, head_dim)
    
    # Create two query batches that are scalar multiples of each other
    q1 = torch.randn(2, 4, 10, head_dim)
    q2 = q1 * 3.14159
    
    # We will pass dummy keys to run the bucketer
    k = torch.randn(2, 4, 10, head_dim)
    
    _, diags1 = bucketer(q1, k, local_window=4, nearby_buckets=1)
    _, diags2 = bucketer(q2, k, local_window=4, nearby_buckets=1)
    
    # The bucket assignments must be identical
    q_buckets_1 = diags1["q_buckets"]
    q_buckets_2 = diags2["q_buckets"]
    
    assert torch.equal(q_buckets_1, q_buckets_2), "Spherical routing is not scale-invariant!"
