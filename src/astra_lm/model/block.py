import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .config import ModelConfig
from .norms import RMSNorm
from .attention_gqa import GroupedQueryAttention
from .chakra_attention import ChakraAttention
from .akasha_memory import AkashaMemoryManager
from .mlp import FeedForward
from .surya_mixer import SuryaMixer
from .indra_phase import IndraPhaseLayer
from .fock_ffn import FockFFN

class DecoderBlock(nn.Module):
    """
    DHRUVA Decoder Block.
    Assembles attention (GQA/CHAKRA), memory (AKASHA), FFN (SwiGLU/FOCK),
    and optional spectral mixing (SURYA) / phase gating (INDRA).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Pre-attention normalization and attention path
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        if config.attention_type == "chakra":
            self.attention = ChakraAttention(config)
        else:
            self.attention = GroupedQueryAttention(config)

        # AKASHA memory manager
        self.memory = AkashaMemoryManager(config)
        
        # Optional spectral mixing global correction
        if config.use_surya:
            self.surya = SuryaMixer(config)

        # Optional real-valued phase/magnitude gate
        if config.use_indra_phase:
            self.indra = IndraPhaseLayer(config)

        # Pre-FFN normalization and MLP path
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(config)
        
        # Optional Chebyshev FFN
        if config.ffn_type == "fock":
            self.ffn = FockFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            cos, sin: Position frequencies [1, 1, seq_len, head_dim]
            attention_mask: Mask for padding
            
        Returns:
            hidden_states: Updated representations [batch, seq_len, d_model]
            diagnostics: Attention/routing information
        """
        # --- 1. Attention Block (Pre-Norm + Attention + Memory + Residual) ---
        normed_hidden_states = self.norm1(hidden_states)
        
        diagnostics = {}
        if self.config.attention_type == "chakra":
            local_attn_out, q, k, v, diagnostics = self.attention(
                hidden_states=normed_hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                return_attn_weights=return_attn_weights,
            )
        else:
            local_attn_out, q, k, v = self.attention(
                hidden_states=normed_hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                return_attn_weights=return_attn_weights,
            )
            
        # AKASHA Memory path
        mixed_attn_out = self.memory(
            hidden_states=normed_hidden_states,
            local_attn_out=local_attn_out,
            q=q,
            k=k,
            v=v,
            attention_module=self.attention
        )
        
        # Format shapes and project back
        batch_size, n_heads, seq_len, head_dim = mixed_attn_out.shape
        attn_out = mixed_attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
        attn_out = self.attention.o_proj(attn_out)
        attn_out = self.attention.resid_dropout(attn_out)
        
        # Optional spectral mixing path
        if self.config.use_surya:
            attn_out = attn_out + self.surya(normed_hidden_states)
            
        hidden_states = hidden_states + attn_out

        # Optional phase/magnitude gate modulation
        if self.config.use_indra_phase:
            hidden_states = self.indra(hidden_states)

        # --- 2. FeedForward Block (Pre-Norm + MLP + Residual) ---
        normed_hidden_states = self.norm2(hidden_states)
        
        if self.config.ffn_type == "fock":
            ffn_out = self.ffn(normed_hidden_states)
        else:
            ffn_out = self.mlp(normed_hidden_states)
            
        hidden_states = hidden_states + ffn_out

        return hidden_states, diagnostics
