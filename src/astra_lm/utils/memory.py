import torch
import gc
import logging

logger = logging.getLogger(__name__)

def log_cuda_memory(label: str = ""):
    """Logs current CUDA memory allocation and reservation in GiB."""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)

    prefix = f"[CUDA MEM] {label}: " if label else "[CUDA MEM]: "
    logger.info(f"{prefix}allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB, peak={peak:.2f} GiB")

def cleanup_memory():
    """Performs aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
