import torch
from astra_lm.model.sphere_bucket import SphereBucketer
import time

def test_memory_efficient_bucketing():
    batch_size = 8
    n_heads = 12
    seq_len = 1024
    head_dim = 64
    num_buckets = 64
    local_window = 256
    nearby_buckets = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

    bucketer = SphereBucketer(num_buckets, head_dim).to(device)

    # Current implementation (may OOM on small GPUs if not careful, but 8*12*1024*1024 is ~100M elements)
    # 100M bools is 100MB. 100M int64 (diff) is 800MB.
    # The OOM in user report: "Tried to allocate 768.00 MiB" matches 8*12*1024*1024 * 8 bytes (int64).

    print("Running bucketer...")
    mask, diagnostics = bucketer(q, k, local_window, nearby_buckets, return_diagnostics=True)
    print(f"Mask shape: {mask.shape}, Candidate ratio: {diagnostics['candidate_ratio']:.2%}")

    assert mask.shape == (batch_size, n_heads, seq_len, seq_len)
    assert mask.dtype == torch.bool

if __name__ == "__main__":
    test_memory_efficient_bucketing()
