import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class SphereBucketer(nn.Module):
    """
    Groups query and key token representations into angular buckets on a hypersphere.
    Allows matching queries with keys belonging to the same or nearby centroids.
    """
    def __init__(self, num_buckets: int, head_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.head_dim = head_dim
        
        # Learnable centroids initialized randomly on unit sphere
        self.centroids = nn.Parameter(torch.randn(num_buckets, head_dim))
        
    def get_unit_centroids(self) -> torch.Tensor:
        return F.normalize(self.centroids, p=2, dim=-1)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        local_window: int, 
        nearby_buckets: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            q: Queries of shape [batch, n_heads, seq_len, head_dim]
            k: Keys of shape [batch, n_heads, seq_len, head_dim]
            local_window: Number of recent tokens to always preserve
            nearby_buckets: Distance in bucket index to include as neighbors
            
        Returns:
            candidate_mask: Boolean tensor of shape [batch, n_heads, seq_len, seq_len]
            diagnostics: Dict containing candidate_ratio, bucket_histogram, etc.
        """
        batch_size, n_heads, seq_len, _ = q.shape
        device = q.device

        # 1. Normalize Q and K to represent directions on the hypersphere
        q_hat = F.normalize(q, p=2, dim=-1)
        k_hat = F.normalize(k, p=2, dim=-1)
        
        # 2. Get normalized centroids
        centroids_unit = self.get_unit_centroids() # [num_buckets, head_dim]
        
        # 3. Assign tokens to buckets based on cosine similarity argmax
        # q_similarities: [batch, n_heads, seq_len, num_buckets]
        q_similarities = torch.matmul(q_hat, centroids_unit.t())
        q_buckets = q_similarities.argmax(dim=-1) # [batch, n_heads, seq_len]
        
        # k_similarities: [batch, n_heads, seq_len, num_buckets]
        k_similarities = torch.matmul(k_hat, centroids_unit.t())
        k_buckets = k_similarities.argmax(dim=-1) # [batch, n_heads, seq_len]
        
        # 4. Construct block candidate mask using modular subtraction
        # diff[b, h, i, j] = (q_buckets[b, h, i] - k_buckets[b, h, j]) % num_buckets
        diff = (q_buckets.unsqueeze(-1) - k_buckets.unsqueeze(-2)) % self.num_buckets
        
        # Select matching and nearby buckets (wrapping around modulo num_buckets)
        bucket_match = (diff <= nearby_buckets) | (diff >= self.num_buckets - nearby_buckets)
        
        # Causal mask constraint (no future leakage)
        causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
        bucket_match = bucket_match & causal_mask.unsqueeze(0).unsqueeze(1)
        
        # 5. Local sliding window mask constraint
        window_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=-local_window + 1)
        local_mask = causal_mask & window_mask
        local_mask_expanded = local_mask.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, seq_len]
        
        # Final candidate mask combines local causal window and bucket routing
        candidate_mask = local_mask_expanded | bucket_match
        
        # 6. Gather diagnostics
        total_elements = batch_size * n_heads * seq_len * seq_len
        candidate_ratio = float(candidate_mask.sum().item()) / total_elements if total_elements > 0 else 1.0
        
        # Calculate bucket histogram across batch/heads/positions
        bucket_hist = torch.bincount(q_buckets.view(-1), minlength=self.num_buckets)
        bucket_histogram = bucket_hist.cpu().tolist()
        
        # Sanity check: local window tokens should be 100% covered
        local_window_coverage = float(((candidate_mask & local_mask_expanded) == local_mask_expanded).all().item())
        
        diagnostics = {
            "candidate_ratio": candidate_ratio,
            "bucket_histogram": bucket_histogram,
            "local_window_coverage": local_window_coverage,
            "q_buckets": q_buckets,
            "k_buckets": k_buckets
        }
        
        return candidate_mask, diagnostics
