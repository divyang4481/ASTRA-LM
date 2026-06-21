import pytest
import torch
import os
import shutil
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.train.checkpoint import save_checkpoint, load_checkpoint
from astra_lm.train.optimizer import create_optimizer
from astra_lm.train.optimizer import get_cosine_schedule_with_warmup

def test_save_load_checkpoint_equivalence():
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
    
    model_1 = DecoderForCausalLM(model_config)
    model_2 = DecoderForCausalLM(model_config)
    
    # Assert models start different
    x = torch.randint(0, model_config.vocab_size, (1, 10))
    with torch.no_grad():
        out_1 = model_1(x)["logits"]
        out_2 = model_2(x)["logits"]
    assert not torch.allclose(out_1, out_2), "Randomly initialized models should not have identical logits"
    
    # Save checkpoint for model_1
    output_dir = "outputs/test_checkpoint"
    optimizer = create_optimizer(model_1, learning_rate=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=10)
    
    checkpoint_path = save_checkpoint(
        output_dir=output_dir,
        step=5,
        model=model_1,
        optimizer=optimizer,
        scheduler=scheduler,
        config=None
    )
    
    # Load checkpoint into model_2
    load_checkpoint(checkpoint_path, model_2)
    
    # Verify logits are identical now
    with torch.no_grad():
        out_1_post = model_1(x)["logits"]
        out_2_post = model_2(x)["logits"]
        
    assert torch.allclose(out_1_post, out_2_post), "Loaded model logits are not identical to saved model logits!"
    
    # Cleanup output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
