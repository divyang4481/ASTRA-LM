import os
import torch
import torch.nn as nn
from typing import Optional
from ..model.decoder import DecoderForCausalLM
from ..model.config import ModelConfig
from ..train.checkpoint import load_checkpoint

def load_teacher_model(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bf16"
) -> nn.Module:
    """
    Loads a teacher model. If config_path exists as a file, loads DecoderForCausalLM from config.
    Otherwise, loads a pre-trained model from Hugging Face (e.g. 'gpt2-medium').
    Supports fp32, fp16, bf16, 8bit, and 4bit loading modes to optimize VRAM.
    """
    import torch
    
    # Map dtype string to torch dtype
    torch_dtype = torch.float32
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16

    if os.path.exists(config_path):
        from ..utils import load_config_from_yaml
        config = load_config_from_yaml(ModelConfig, config_path)
        model = DecoderForCausalLM(config)
        if torch_dtype != torch.float32:
            model = model.to(torch_dtype)
        
        if checkpoint_path:
            load_checkpoint(checkpoint_path, model, map_location=device)
        model.to(device)
    else:
        from transformers import GPT2LMHeadModel
        print(f"Loading pretrained teacher from Hugging Face: {config_path} with dtype/quant: {dtype}")
        
        if dtype in ("8bit", "4bit"):
            try:
                import bitsandbytes
                import accelerate
            except ImportError:
                print("Warning: bitsandbytes or accelerate not found. Falling back to bfloat16 for teacher model loading.")
                dtype = "bf16"
                torch_dtype = torch.bfloat16
                
        if dtype == "8bit":
            model = GPT2LMHeadModel.from_pretrained(
                config_path,
                load_in_8bit=True,
                device_map="auto" if device == "cuda" else None
            )
        elif dtype == "4bit":
            model = GPT2LMHeadModel.from_pretrained(
                config_path,
                load_in_4bit=True,
                device_map="auto" if device == "cuda" else None
            )
        else:
            model = GPT2LMHeadModel.from_pretrained(
                config_path,
                torch_dtype=torch_dtype
            )
            model.to(device)
        
    model.eval() # Teacher always in eval mode
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model
