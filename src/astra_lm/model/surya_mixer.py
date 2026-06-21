import torch
import torch.nn as nn
from .config import ModelConfig

class SuryaMixer(nn.Module):
    """
    Periodic global spectral correction layer.
    Mixes sequence representations in the frequency domain using FFT.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Learnable filter weights for the sequence length
        self.filter_weight = nn.Parameter(torch.randn(config.max_seq_len, config.d_model))
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        v = self.w_v(x)
        
        # 1D FFT along the sequence dimension
        v_fft = torch.fft.fft(v, dim=1)
        
        # Slice filter weights to the current sequence length and multiply
        w = self.filter_weight[:seq_len, :].unsqueeze(0) # [1, seq_len, d_model]
        out_fft = v_fft * w
        
        # Inverse FFT
        out = torch.fft.ifft(out_fft, dim=1).real
        
        # Output projection
        return self.w_o(out)
