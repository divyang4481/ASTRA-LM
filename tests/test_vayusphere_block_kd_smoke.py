import torch
import torch.nn as nn
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.train.kd_trainer import KDTrainer
from astra_lm.train.config import TrainConfig
from torch.utils.data import DataLoader
from astra_lm.data.dataset import SyntheticDataset
from astra_lm.data.collator import CausalLMCollator

def test_kd_smoke_cpu_teacher():
    # Tiny student
    s_config = ModelConfig(
        vocab_size=1000, max_seq_len=64, d_model=64, n_layers=1, n_heads=2, n_kv_heads=2,
        attention_impl="vayusphere_block", vayu_block_size=16, vayu_top_m_blocks=2
    )
    student = DecoderForCausalLM(s_config)

    # Tiny teacher
    t_config = ModelConfig(
        vocab_size=1000, max_seq_len=64, d_model=64, n_layers=1, n_heads=2, n_kv_heads=2
    )
    teacher = DecoderForCausalLM(t_config).to("cpu").eval()
    for param in teacher.parameters():
        param.requires_grad = False

    train_config = TrainConfig(
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        output_dir="outputs/test_kd_smoke",
        overwrite_output_dir=True
    )

    dataset = SyntheticDataset(vocab_size=1000, seq_len=64, num_samples=2)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=CausalLMCollator())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = student.to(device)

    trainer = KDTrainer(
        student_model=student,
        teacher_model=teacher,
        train_config=train_config,
        train_dataloader=dataloader,
        device=device,
        topk_logits=10
    )

    # Capture initial weights
    initial_weights = student.model.blocks[0].attention.q_proj.weight.clone()

    trainer.train()

    # Check if weights updated
    updated_weights = student.model.blocks[0].attention.q_proj.weight
    assert not torch.equal(initial_weights, updated_weights), "Student weights did not update"

    # Check if teacher weights are same (frozen)
    # teacher is on cpu
    assert teacher.model.blocks[0].attention.q_proj.weight.requires_grad == False
