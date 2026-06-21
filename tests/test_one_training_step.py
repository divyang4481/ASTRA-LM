import pytest
import torch
from torch.utils.data import DataLoader
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.train.config import TrainConfig
from astra_lm.train.trainer import Trainer
from astra_lm.data.dataset import SyntheticDataset
from astra_lm.data.collator import CausalLMCollator

def test_training_step_executes():
    model_config = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
        norm_eps=1e-5,
        bias=False,
        tie_word_embeddings=True,
        attention_type="chakra",
        local_window=8,
        sphere_buckets=4,
        nearby_buckets=1
    )
    
    train_config = TrainConfig(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-3,
        max_steps=2, # Run 2 steps
        warmup_steps=1,
        eval_steps=10,
        save_steps=10,
        logging_steps=1,
        output_dir="outputs/test_train"
    )
    
    model = DecoderForCausalLM(model_config)
    dataset = SyntheticDataset(
        vocab_size=model_config.vocab_size,
        seq_len=model_config.max_seq_len,
        num_samples=10
    )
    
    collator = CausalLMCollator()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    
    trainer = Trainer(
        model=model,
        train_config=train_config,
        train_dataloader=dataloader,
        eval_dataloader=None,
        device="cpu"
    )
    
    # Save initial weights
    initial_weights = model.lm_head.weight.clone()
    
    # Run trainer
    trainer.train()
    
    # Assert weights changed
    assert not torch.equal(initial_weights, model.lm_head.weight), "Model weights did not change after optimizer step!"
