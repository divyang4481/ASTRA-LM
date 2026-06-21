import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: Tuple[float, float],
    eps: float,
) -> torch.optim.Optimizer:
    """
    Creates an AdamW optimizer for the given model, separating parameters
    that should and should not undergo weight decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() < 2 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps
    )

    return optimizer

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, progress)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch)
