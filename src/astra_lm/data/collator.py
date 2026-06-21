import torch
from typing import List, Dict

class CausalLMCollator:
    """
    Collator for causal language modeling.
    """
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        lengths = [len(ex) for ex in examples]
        max_len = max(lengths)

        if all(length == max_len for length in lengths):
            input_ids = torch.stack(examples)
            labels = input_ids.clone()
        else:
            input_ids = torch.full((len(examples), max_len), self.pad_token_id, dtype=torch.long)
            labels = torch.full((len(examples), max_len), -100, dtype=torch.long)

            for i, ex in enumerate(examples):
                input_ids[i, :len(ex)] = ex
                labels[i, :len(ex)] = ex

        return {
            "input_ids": input_ids,
            "labels": labels
        }
