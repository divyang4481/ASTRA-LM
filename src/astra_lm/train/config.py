from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    # Batch sizes
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4

    # Optimizer settings
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Scheduler settings
    max_steps: int = 1000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1

    # Training Loop
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    max_eval_batches: int = 50

    # Directories
    output_dir: str = "outputs"

@dataclass
class DataConfig:
    seq_len: int = 128
    vocab_size: int = 32000

    # For synthetic dataset
    dataset_type: str = "synthetic"  # "synthetic" or "pretokenized"
    num_samples: int = 1000

    # For real datasets
    data_path: Optional[str] = None
