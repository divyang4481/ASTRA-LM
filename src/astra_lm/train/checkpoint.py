import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(
    output_dir: str,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: Any,
    filename_prefix: str = "checkpoint"
) -> str:
    """
    Saves a training checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{filename_prefix}-{step}.pt")

    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "step": step,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    prune_checkpoints(output_dir, filename_prefix, keep_latest=2)

    return checkpoint_path

def prune_checkpoints(output_dir: str, prefix: str, keep_latest: int = 2):
    """
    Deletes old checkpoints, keeping only the most recent N ones.
    """
    checkpoints = [
        f for f in os.listdir(output_dir) 
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    
    def extract_step(filename):
        try:
            return int(filename.split("-")[-1].split(".")[0])
        except ValueError:
            return -1

    checkpoints.sort(key=extract_step)

    if len(checkpoints) > keep_latest:
        to_remove = checkpoints[:-keep_latest]
        for f in to_remove:
            path = os.path.join(output_dir, f)
            try:
                os.remove(path)
                logger.info(f"Pruned old checkpoint: {f}")
            except Exception as e:
                logger.warning(f"Failed to prune {f}: {e}")

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    map_location: str = "cpu"
) -> Dict[str, Any]:
    """
    Loads a training checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    model_state = checkpoint["model"]
    if hasattr(model, "module") and not list(model_state.keys())[0].startswith("module."):
        model.module.load_state_dict(model_state)
    elif not hasattr(model, "module") and list(model_state.keys())[0].startswith("module."):
        model_state = {k[7:]: v for k, v in model_state.items()}
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
