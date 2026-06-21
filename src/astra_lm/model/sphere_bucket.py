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
        nearby_buckets: int,
        chunk_size: int = 512,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            q: Queries of shape [batch, n_heads, seq_len, head_dim]
            k: Keys of shape [batch, n_heads, seq_len, head_dim]
            local_window: Number of recent tokens to always preserve
            nearby_buckets: Distance in bucket index to include as neighbors
            chunk_size: Processing chunk size for memory efficiency
            
        Returns:
            candidate_mask: Boolean tensor of shape [batch, n_heads, seq_len, seq_len]
            diagnostics: Dict containing candidate_ratio, bucket_histogram, etc.
        """
        batch_size, n_heads, seq_len, _ = q.shape
        device = q.device

        # 1. Normalize Q and K to represent directions on the hypersphere
        # Use autocast context if available to save memory during normalization/matmul
        q_hat = F.normalize(q, p=2, dim=-1)
        k_hat = F.normalize(k, p=2, dim=-1)
        
        # 2. Get normalized centroids
        centroids_unit = self.get_unit_centroids() # [num_buckets, head_dim]
        
        # 3. Assign tokens to buckets based on cosine similarity argmax
        # q_similarities: [batch, n_heads, seq_len, num_buckets]
        q_similarities = torch.matmul(q_hat, centroids_unit.t())
        q_buckets = q_similarities.argmax(dim=-1).to(torch.int16) # [batch, n_heads, seq_len]
        
        # k_similarities: [batch, n_heads, seq_len, num_buckets]
        k_similarities = torch.matmul(k_hat, centroids_unit.t())
        k_buckets = k_similarities.argmax(dim=-1).to(torch.int16) # [batch, n_heads, seq_len]
        
        # 4. Construct candidate mask efficiently
        # We avoid (q_buckets.unsqueeze(-1) - k_buckets.unsqueeze(-2)) % num_buckets
        # because it materializes a [B, H, S, S] int64 tensor (8 bytes per element).
        # Boolean mask is only 1 byte per element.
        
        candidate_mask = torch.zeros((batch_size, n_heads, seq_len, seq_len), dtype=torch.bool, device=device)
        
        # Pre-calculate local window mask to avoid re-calculating in chunks
        causal_mask_base = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
        window_mask_base = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=-local_window + 1)
        local_mask = causal_mask_base & window_mask_base
        
        # Process in chunks of queries to save peak memory
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_buckets_chunk = q_buckets[:, :, i:end_i].unsqueeze(-1) # [B, H, chunk, 1]

            # Broadcast against all keys
            # Use int16 for modular arithmetic to save memory if it were to materialize temporarily,
            # though here it's still [B, H, chunk, S]
            diff_chunk = (q_buckets_chunk - k_buckets.unsqueeze(-2)) % self.num_buckets

            # Bucket match for this chunk
            match_chunk = (diff_chunk <= nearby_buckets) | (diff_chunk >= self.num_buckets - nearby_buckets)

            # Apply causal constraint for this chunk
            # causal_mask_chunk[i:end_i, :]
            match_chunk &= causal_mask_base[i:end_i, :].unsqueeze(0).unsqueeze(1)

            # Combine with local window
            candidate_mask[:, :, i:end_i, :] = local_mask[i:end_i, :].unsqueeze(0).unsqueeze(1) | match_chunk
        
        # 6. Gather diagnostics (Optional and gated for performance)
        diagnostics = {}
        if return_diagnostics:
            total_elements = batch_size * n_heads * seq_len * seq_len
            candidate_ratio = float(candidate_mask.sum().item()) / total_elements if total_elements > 0 else 1.0

            # Calculate bucket histogram across batch/heads/positions
            # q_buckets is [B, H, S] int16
            bucket_hist = torch.bincount(q_buckets.reshape(-1).to(torch.long), minlength=self.num_buckets)
            bucket_histogram = bucket_hist.cpu().tolist()

            # Sanity check: local window tokens should be 100% covered
            local_mask_expanded = local_mask.unsqueeze(0).unsqueeze(1)
            local_window_coverage = float(((candidate_mask & local_mask_expanded) == local_mask_expanded).all().item())

            diagnostics = {
                "candidate_ratio": candidate_ratio,
                "bucket_histogram": bucket_histogram,
                "local_window_coverage": local_window_coverage,
                "q_buckets": q_buckets,
                "k_buckets": k_buckets
            }
        
        return candidate_mask, diagnostics
