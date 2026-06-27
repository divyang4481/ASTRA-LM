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
    import logging
    logger = logging.getLogger(__name__)

    # Handle auto device
    actual_device = device
    if device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
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
            load_checkpoint(checkpoint_path, model, map_location=actual_device)

        try:
            model.to(actual_device)
        except torch.cuda.OutOfMemoryError:
            if device == "auto" and actual_device == "cuda":
                logger.warning("Teacher model failed to load on CUDA (OOM). Falling back to CPU.")
                model.to("cpu")
            else:
                raise
    else:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loading pretrained teacher from Hugging Face: {config_path} with dtype/quant: {dtype}")
        
        if dtype in ("8bit", "4bit"):
            try:
                import bitsandbytes
                import accelerate
            except ImportError:
                logger.warning(f"bitsandbytes or accelerate not found. Falling back from {dtype} to bfloat16 for teacher model loading.")
                dtype = "bf16"
                torch_dtype = torch.bfloat16
                
        if dtype == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                config_path,
                quantization_config=bnb_config,
                device_map="auto" if actual_device == "cuda" else None
            )
        elif dtype == "4bit":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                config_path,
                quantization_config=bnb_config,
                device_map="auto" if actual_device == "cuda" else None
            )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    config_path,
                    torch_dtype=torch_dtype
                )
                if not hasattr(model, "hf_device_map"): # Don't move if using device_map
                    model.to(actual_device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if (device == "auto" or device == "cpu") and "out of memory" in str(e).lower():
                     logger.warning(f"Teacher model failed to load on {actual_device} ({e}). Falling back to CPU.")
                     model = AutoModelForCausalLM.from_pretrained(
                        config_path,
                        torch_dtype=torch_dtype
                     ).to("cpu")
                else:
                    raise
        
    model.eval() # Teacher always in eval mode
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model
