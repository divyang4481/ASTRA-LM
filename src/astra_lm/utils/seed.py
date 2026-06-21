import torch
import random
import numpy as np
import os

def set_seed(seed: int):
    """
    Sets the seed for reproducibility across python, numpy, torch, and cuda.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in some torch operations
    # Note: This can impact performance slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For newer versions of torch
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to: {seed}")
