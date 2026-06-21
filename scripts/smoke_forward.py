import torch
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM

def main():
    print("Initializing tiny config for smoke test...")
    config = ModelConfig(
        vocab_size=128,
        max_seq_len=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        rope_base=10000.0,
        norm_eps=1e-5,
        bias=False,
        tie_word_embeddings=True,
        attention_type="chakra",
        local_window=16,
        sphere_buckets=8,
        nearby_buckets=1
    )

    print("Instantiating DecoderForCausalLM...")
    model = DecoderForCausalLM(config)

    batch_size = 2
    seq_len = 10
    print(f"Creating random input IDs with shape ({batch_size}, {seq_len})...")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    print("Running forward pass...")
    outputs = model(input_ids, labels=labels)

    print("\n--- Outputs ---")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    if "diagnostics" in outputs:
        print("\nCHAKRA Routing Diagnostics:")
        for idx, diag in enumerate(outputs["diagnostics"]):
            ratio = diag.get('candidate_ratio', 1.0)
            print(f"  Layer {idx}: Candidate Ratio = {ratio:.2%}")
            if "bucket_histogram" in diag:
                print(f"           Bucket Histogram = {diag['bucket_histogram']}")

    print("\nSmoke test passed successfully!")

if __name__ == "__main__":
    main()
