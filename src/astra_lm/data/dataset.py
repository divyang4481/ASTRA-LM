import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Union
from pathlib import Path

class SyntheticDataset(Dataset):
    """
    A simple synthetic dataset that generates random token sequences.
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)

class PretokenizedDataset(Dataset):
    """
    A dataset that loads pre-tokenized integer sequences from a numpy array,
    file path (via mmap), or list of lists and provides fixed-length chunks.
    """
    def __init__(self, data: Union[str, Path, np.ndarray, List[List[int]], torch.Tensor], seq_len: int):
        self.seq_len = seq_len

        if isinstance(data, (str, Path)):
            self.data = np.load(str(data), mmap_mode='r')
            if self.data.ndim > 1:
                self.data = self.data.reshape(-1)
        elif isinstance(data, list):
            if data and isinstance(data[0], list):
                flat_data = [item for sublist in data for item in sublist]
                self.data = np.array(flat_data, dtype=np.int64)
            else:
                self.data = np.array(data, dtype=np.int64)
        elif isinstance(data, np.ndarray):
            self.data = data
            if self.data.ndim > 1:
                self.data = self.data.reshape(-1)
        elif isinstance(data, torch.Tensor):
            self.data = data
            if self.data.dim() > 1:
                self.data = self.data.view(-1)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        self.num_samples = len(self.data) // self.seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        chunk = self.data[start_idx:end_idx]

        if isinstance(chunk, torch.Tensor):
            return chunk.long()
        else:
            return torch.tensor(np.array(chunk), dtype=torch.long)
