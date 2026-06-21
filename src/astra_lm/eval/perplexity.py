import torch
import torch.nn as nn
from typing import Dict, Optional
import math
from tqdm import tqdm

def calculate_perplexity(mean_loss: float) -> float:
    """
    Calculates perplexity from mean cross entropy loss.
    """
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return float('inf')

def evaluate_perplexity(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluates a model's perplexity on a given dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    num_batches = len(dataloader)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)

    with torch.no_grad():
        pbar = tqdm(total=num_batches, desc="Evaluating", leave=False)
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
                
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device) if "labels" in batch else input_ids

            outputs = model(input_ids=input_ids, labels=labels)

            if "loss" in outputs:
                total_loss += outputs["loss"].item()
                total_batches += 1
            
            pbar.update(1)
        pbar.close()

    if total_batches == 0:
        return {"loss": 0.0, "perplexity": 0.0}

    mean_loss = total_loss / total_batches
    perplexity = calculate_perplexity(mean_loss)

    return {
        "loss": mean_loss,
        "perplexity": perplexity
    }
