import torch
import torch.nn.functional as F
from typing import Tuple, Dict

def build_vayusphere_block_routes(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int,
    top_m_blocks: int,
    force_local_blocks: bool = True,
    route_policy: str = "current_prev_semantic",
) -> Tuple[torch.Tensor, Dict]:
    """
    Builds block-level routes for VayuSphere attention.

    Args:
        q, k: [batch, heads, seq_len, head_dim]
        block_size: Number of tokens per block
        top_m_blocks: Number of key blocks to route each query block to
        force_local_blocks: If True, always include current and previous blocks
        route_policy: Routing strategy ("semantic_only", "current_prev_semantic", "current_prev_only")

    Returns:
        route_idx: [batch, heads, num_blocks, top_m_blocks], int32
        stats: Dictionary of routing statistics
    """
    batch_size, n_heads, seq_len, head_dim = q.shape
    num_blocks = seq_len // block_size
    assert seq_len % block_size == 0, f"seq_len ({seq_len}) must be divisible by block_size ({block_size})"

    # 1. Normalize q/k onto hypersphere
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)

    # 2. Compute block centroids
    # [batch, heads, num_blocks, block_size, head_dim]
    q_blocks = q_norm.view(batch_size, n_heads, num_blocks, block_size, head_dim)
    k_blocks = k_norm.view(batch_size, n_heads, num_blocks, block_size, head_dim)

    # mean inside block
    q_centroids = q_blocks.mean(dim=3) # [batch, heads, num_blocks, head_dim]
    k_centroids = k_blocks.mean(dim=3) # [batch, heads, num_blocks, head_dim]

    # 3. Normalize centroids
    q_centroids = F.normalize(q_centroids, p=2, dim=-1)
    k_centroids = F.normalize(k_centroids, p=2, dim=-1)

    # 4. Compute block cosine similarity
    # [batch, heads, num_blocks, num_blocks]
    block_scores = torch.matmul(q_centroids, k_centroids.transpose(-1, -2))

    # 5. Apply causal block mask
    # A query block i can only see key blocks j where j <= i
    causal_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=q.device, dtype=torch.bool))
    block_scores = block_scores.masked_fill(~causal_mask, -float('inf'))

    # 6 & 7. Routing policy
    # route_idx: [batch, heads, num_blocks, top_m_blocks]
    route_idx = torch.full((batch_size, n_heads, num_blocks, top_m_blocks), -1, device=q.device, dtype=torch.int32)

    for i in range(num_blocks):
        # Indices of valid key blocks (causal)
        valid_indices = torch.arange(i + 1, device=q.device, dtype=torch.int32)

        selected_indices = []

        if route_policy in ["current_prev_semantic", "current_prev_only"] and force_local_blocks:
            # Always include current block
            selected_indices.append(i)
            # Always include previous block when available
            if i > 0:
                selected_indices.append(i - 1)

        if route_policy != "current_prev_only":
            # Fill remaining slots with best older semantic blocks
            needed = top_m_blocks - len(selected_indices)
            if needed > 0:
                scores_i = block_scores[:, :, i, :i+1].clone()
                # Mask already selected to avoid duplicates
                for idx in selected_indices:
                    scores_i[:, :, idx] = -float('inf')

                # topk over valid indices
                # We need to handle cases where there are fewer than 'needed' valid blocks left
                actual_needed = min(needed, scores_i.size(-1))
                if actual_needed > 0:
                    _, topk_idx = torch.topk(scores_i, k=actual_needed, dim=-1)
                    # topk_idx: [batch, heads, actual_needed]
                    # This is a bit tricky to batch properly without loops if we want efficiency,
                    # but for v0 PyTorch implementation, simplicity is fine.
                    # Actually, we can just use the topk_idx.

                    # For simplicity in v0, let's just do it per batch/head if needed,
                    # but actually topk works fine on the last dim.
                    # However, merging with selected_indices is the hard part in a vectorized way.
                    pass

        # Redoing the loop properly for selection logic to handle varying lengths and duplicates

    # Revised selection logic (vectorized-ish where possible, but safe)
    # 1. Identify "local" indices
    local_indices = []
    if route_policy in ["current_prev_semantic", "current_prev_only"] and force_local_blocks:
        # We'll use a mask-based approach
        pass

    # Actually, let's use a simpler approach for v0:
    # 1. Start with scores
    # 2. Add a large value to "forced" blocks
    # 3. Take top-m

    modified_scores = block_scores.clone()
    if route_policy in ["current_prev_semantic", "current_prev_only"] and force_local_blocks:
        # Add a large constant to current and previous block scores to ensure they are picked
        # current blocks (j=i)
        curr_idx = torch.arange(num_blocks, device=q.device)
        modified_scores[:, :, curr_idx, curr_idx] += 100.0
        # previous blocks (j=i-1)
        prev_idx = torch.arange(1, num_blocks, device=q.device)
        modified_scores[:, :, prev_idx, prev_idx - 1] += 100.0

    # 4. Handle early blocks where j <= i has fewer than top_m_blocks
    # Actually, topk will just take what it can if we are careful,
    # but the problem says "duplicate the current or previous valid block".

    # We can fill -inf with a very small number but valid index to allow topk to pick something
    # but better to just clamp/repeat.

    # Let's use a different strategy:
    # For each query block i, we look at j in [0, i].
    # There are i+1 candidates.
    # We want to pick top_m.
    # If i+1 < top_m, we must duplicate.

    # To handle this vectorized:
    # Use a large constant for forced blocks.
    # For j > i, scores are -inf.
    # To handle i+1 < top_m, we can replace -inf with a slightly larger but still small value for j=i (self-loop).

    # Ensure current block is always at least reachable
    for i in range(num_blocks):
        modified_scores[:, :, i, i] = torch.where(
            modified_scores[:, :, i, i] == -float('inf'),
            torch.tensor(100.0, device=q.device), # Should not happen due to causal mask i,i is allowed
            modified_scores[:, :, i, i]
        )

    # top_m_blocks
    # If num_blocks < top_m_blocks, topk will fail.
    actual_top_m = min(top_m_blocks, num_blocks)
    _, route_idx_small = torch.topk(modified_scores, k=actual_top_m, dim=-1)

    if actual_top_m < top_m_blocks:
        # Duplicate the last selected (which would be the current block i due to the +100)
        # padding route_idx to top_m_blocks
        padding = route_idx_small[:, :, :, -1:].expand(-1, -1, -1, top_m_blocks - actual_top_m)
        route_idx = torch.cat([route_idx_small, padding], dim=-1)
    else:
        route_idx = route_idx_small

    # 8. Clamp route indices so no future block appears (already handled by causal mask + topk)
    # But for safety:
    curr_block_idx = torch.arange(num_blocks, device=q.device, dtype=torch.int32).view(1, 1, num_blocks, 1)
    route_idx = torch.minimum(route_idx, curr_block_idx)

    # Stats
    with torch.no_grad():
        dense_pairs = (num_blocks * (num_blocks + 1)) / 2 * block_size * block_size
        active_pairs_est = num_blocks * top_m_blocks * block_size * block_size
        pair_reduction_est = active_pairs_est / dense_pairs if dense_pairs > 0 else 1.0

        stats = {
            "num_blocks": num_blocks,
            "top_m_blocks": top_m_blocks,
            "dense_pairs": int(dense_pairs),
            "active_pairs_est": int(active_pairs_est),
            "pair_reduction_est": float(pair_reduction_est)
        }

    return route_idx.to(torch.int32), stats
