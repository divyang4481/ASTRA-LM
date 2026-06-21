import torch
from astra_lm.model.sphere_bucket import SphereBucketer

def old_forward(self, q, k, local_window, nearby_buckets):
    batch_size, n_heads, seq_len, _ = q.shape
    device = q.device
    q_hat = torch.nn.functional.normalize(q, p=2, dim=-1)
    k_hat = torch.nn.functional.normalize(k, p=2, dim=-1)
    centroids_unit = self.get_unit_centroids()
    q_similarities = torch.matmul(q_hat, centroids_unit.t())
    q_buckets = q_similarities.argmax(dim=-1)
    k_similarities = torch.matmul(k_hat, centroids_unit.t())
    k_buckets = k_similarities.argmax(dim=-1)
    diff = (q_buckets.unsqueeze(-1) - k_buckets.unsqueeze(-2)) % self.num_buckets
    bucket_match = (diff <= nearby_buckets) | (diff >= self.num_buckets - nearby_buckets)
    causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
    bucket_match = bucket_match & causal_mask.unsqueeze(0).unsqueeze(1)
    window_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=-local_window + 1)
    local_mask = causal_mask & window_mask
    local_mask_expanded = local_mask.unsqueeze(0).unsqueeze(1)
    candidate_mask = local_mask_expanded | bucket_match
    return candidate_mask

def test_correctness():
    batch_size = 2
    n_heads = 4
    seq_len = 128
    head_dim = 32
    num_buckets = 16
    local_window = 32
    nearby_buckets = 1

    device = "cpu"
    torch.manual_seed(42)

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

    bucketer = SphereBucketer(num_buckets, head_dim).to(device)

    # New efficient forward
    mask_new, _ = bucketer(q, k, local_window, nearby_buckets, chunk_size=32)

    # Old dense forward
    mask_old = old_forward(bucketer, q, k, local_window, nearby_buckets)

    diff = (mask_new != mask_old).sum().item()
    print(f"Differences: {diff}")
    assert diff == 0, f"Found {diff} differences between old and new bucketing logic!"
    print("Correctness test passed!")

if __name__ == "__main__":
    test_correctness()
