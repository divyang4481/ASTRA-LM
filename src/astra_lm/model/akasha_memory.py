import torch
import torch.nn as nn
from typing import Tuple
from .config import ModelConfig

class RecentMemoryBank(nn.Module):
    """
    Standard sliding-window memory bank for local context.
    """
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return k, v

    def get_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        window_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-self.window_size).bool()
        return mask | window_mask

class AnchorMemoryBank(nn.Module):
    """
    Selectively preserves 'anchor' tokens from the distant past.
    Keep BOS token + every N-th token.
    """
    def __init__(self, anchor_interval: int = 16):
        super().__init__()
        self.anchor_interval = anchor_interval

    def select_anchors(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = k.size(-2)
        if seq_len < self.anchor_interval:
            return k, v

        indices_list = [0]
        for i in range(self.anchor_interval, seq_len, self.anchor_interval):
            indices_list.append(i)
        
        indices = torch.tensor(indices_list, device=k.device, dtype=torch.long)
        
        k_anchors = k.index_select(-2, indices)
        v_anchors = v.index_select(-2, indices)
        
        return k_anchors, v_anchors

class AkashaMemoryManager(nn.Module):
    """
    Orchestrates the AKASHA memory banks and gated mixing.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.recent_bank = RecentMemoryBank(config.memory_window)
        self.anchor_bank = AnchorMemoryBank(config.anchor_interval)
        
        # Gated mixing projection
        self.gate = nn.Linear(config.d_model, 1)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        local_attn_out: torch.Tensor, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        attention_module: nn.Module
    ) -> torch.Tensor:
        """
        Mixes local attention output with anchor memory attention.
        """
        # If anchor memory is disabled, bypass and return local attention directly
        if not self.config.use_anchor_bank:
            return local_attn_out

        # Select anchors from K/V states
        k_anchors, v_anchors = self.anchor_bank.select_anchors(k, v)
        k_anchors, v_anchors = attention_module.expand_kv(k_anchors, v_anchors)
        
        # Attention over distant anchors
        if hasattr(attention_module, "scaled_dot_product_attention_masked"):
            # For ChakraAttention, use unmasked dot product on anchor dimensions
            batch, heads, seq_len, _ = q.shape
            anchor_len = k_anchors.size(-2)
            dummy_mask = torch.ones((batch, heads, seq_len, anchor_len), dtype=torch.bool, device=q.device)
            anchor_attn_out = attention_module.scaled_dot_product_attention_masked(
                q, k_anchors, v_anchors, mask=dummy_mask
            )
        else:
            anchor_attn_out = attention_module.scaled_dot_product_attention(
                q, k_anchors, v_anchors, mask=None
            )
            
        # Compute gate value
        g = torch.sigmoid(self.gate(hidden_states)).transpose(1, 2).unsqueeze(-1)
        
        # Mix local context and distant anchor memory
        mixed_out = g * local_attn_out + (1 - g) * anchor_attn_out
        
        return mixed_out
