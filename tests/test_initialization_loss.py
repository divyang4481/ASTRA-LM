import torch
import math
import pytest
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.model.config import ModelConfig

def test_initialization_loss():
    vocab_size = 128
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        tie_word_embeddings=True
    )

    model = DecoderForCausalLM(config)
    model.eval()

    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    logits = outputs["logits"]
    loss = outputs["loss"].item()

    expected_loss = math.log(vocab_size)
    print(f"Observed loss: {loss:.4f}")
    print(f"Expected loss (ln({vocab_size})): {expected_loss:.4f}")

    # Assert logits are finite
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"

    # Assert loss is reasonably close to ln(vocab_size)
    # Range suggested: 3.5 < loss < 6.5 for ln(128) approx 4.85
    assert 3.5 < loss < 6.5, f"Initial loss {loss:.4f} is too far from ln({vocab_size}) = {expected_loss:.4f}"

if __name__ == "__main__":
    test_initialization_loss()
