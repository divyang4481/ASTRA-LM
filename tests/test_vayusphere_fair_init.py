import torch
import copy
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM

def test_fair_init_baseline_vs_vayusphere():
    seed = 42

    # 1. Create baseline model with seed
    torch.manual_seed(seed)
    base_cfg = ModelConfig(
        vocab_size=128,
        max_seq_len=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        use_vayusphere=False
    )
    baseline = DecoderForCausalLM(base_cfg)
    baseline_sd = baseline.state_dict()

    # 2. Create VayuSphere model with SAME seed
    torch.manual_seed(seed)
    vs_cfg = copy.deepcopy(base_cfg)
    vs_cfg.use_vayusphere = True
    vs_cfg.vayusphere_num_centroids = 8
    vs_cfg.vayusphere_target = "qk"

    vayusphere = DecoderForCausalLM(vs_cfg)
    vs_sd_before = vayusphere.state_dict()

    # Verify that even with same seed, weights might differ due to VayuSphere init shifting RNG
    # (This is what we want to fix/ensure we handle by loading state dict)

    # 3. Load baseline state dict into VayuSphere model
    missing, unexpected = vayusphere.load_state_dict(baseline_sd, strict=False)

    # Assertions
    assert not unexpected, f"Unexpected keys: {unexpected}"

    # Missing keys should only be VayuSphere specific (centroids)
    for key in missing:
        assert "vayusphere" in key or "centroids" in key, f"Non-VayuSphere key missing: {key}"

    vs_sd_after = vayusphere.state_dict()

    # 4. Verify all common parameters match exactly
    for name, tensor in baseline_sd.items():
        assert name in vs_sd_after, f"Key {name} missing in VayuSphere model after loading"
        torch.testing.assert_close(tensor, vs_sd_after[name], msg=f"Weight mismatch for {name}")

    print("Fair initialization test passed!")

if __name__ == "__main__":
    test_fair_init_baseline_vs_vayusphere()
