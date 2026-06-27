import torch
import torch.nn as nn
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.train.kd_trainer import KDTrainer
from astra_lm.train.config import TrainConfig
import pytest

class MockDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, vocab_size):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    def __len__(self):
        return 10
    def __getitem__(self, idx):
        return {"input_ids": torch.randint(0, self.vocab_size, (self.seq_len,))}

def test_kd_trainer_smoke():
    vocab_size = 100
    seq_len = 32
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
    )

    student = DecoderForCausalLM(config)
    teacher = DecoderForCausalLM(config)

    train_config = TrainConfig(
        output_dir="tmp_kd",
        max_steps=2,
        logging_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        overwrite_output_dir=True
    )

    dataset = MockDataset(seq_len, vocab_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    trainer = KDTrainer(
        student_model=student,
        teacher_model=teacher,
        train_config=train_config,
        train_dataloader=dataloader,
        alpha=0.5,
        temperature=2.0,
        device="cpu", # Force CPU for smoke test
        topk_logits=10
    )

    trainer.train()
    assert True # If it finishes without error, it's fine

def test_kl_topk_edge_cases():
    from astra_lm.distill.kd_losses import kl_topk_distillation_loss

    batch, seq, vocab = 2, 8, 100
    student_logits = torch.randn(batch, seq, vocab)

    # top-k = vocab (full)
    k = vocab
    values, indices = torch.topk(student_logits, k=k)
    loss_full = kl_topk_distillation_loss(student_logits, indices, values)
    assert not torch.isnan(loss_full)

    # top-k = 1
    k = 1
    values, indices = torch.topk(student_logits, k=k)
    loss_1 = kl_topk_distillation_loss(student_logits, indices, values)
    assert not torch.isnan(loss_1)

if __name__ == "__main__":
    test_kd_trainer_smoke()
    test_kl_topk_edge_cases()
    print("KD Smoke tests passed!")
