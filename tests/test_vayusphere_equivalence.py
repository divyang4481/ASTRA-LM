import torch
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM

def test_vayusphere_alpha_zero_equivalence():
    config_base = ModelConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        use_vayusphere=False,
        attention_type="sdpa"
    )

    config_vs = ModelConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        use_vayusphere=True,
        vayusphere_alpha=0.0,
        attention_type="sdpa"
    )

    torch.manual_seed(42)
    model_base = DecoderForCausalLM(config_base).eval()

    torch.manual_seed(42)
    model_vs = DecoderForCausalLM(config_vs).eval()

    # Verify weights are same
    for p1, p2 in zip(model_base.parameters(), model_vs.parameters()):
        # Note: model_vs has extra centroids parameters.
        # But for same named parameters they should match if seed was same and init order was same.
        # Actually centroids init will shift the RNG.
        pass

    # Let's manually copy state dict for all except centroids
    state_dict = model_base.state_dict()
    model_vs.load_state_dict(state_dict, strict=False)

    x = torch.randint(0, 100, (1, 10))

    with torch.no_grad():
        out_base = model_base(x)
        out_vs = model_vs(x)

    assert torch.allclose(out_base["logits"], out_vs["logits"], atol=1e-6)
    print("VayuSphere alpha=0 equivalence test passed!")

if __name__ == "__main__":
    test_vayusphere_alpha_zero_equivalence()
