import torch.nn as nn
from typing import Optional
from ..model.decoder import DecoderForCausalLM
from ..model.config import ModelConfig
from ..train.checkpoint import load_checkpoint

def load_teacher_model(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
) -> nn.Module:
    """
    Loads a teacher model from a config and optional checkpoint.
    """
    from ..utils import load_config_from_yaml
    
    config = load_config_from_yaml(ModelConfig, config_path)
    model = DecoderForCausalLM(config)
    
    if checkpoint_path:
        load_checkpoint(checkpoint_path, model, map_location=device)
    
    model.to(device)
    model.eval() # Teacher always in eval mode
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model
