import argparse
import torch
import time
import logging
import numpy as np
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.utils import load_config_from_yaml

def benchmark(model, input_ids, label="Model", warmups=5, steps=10, device="cuda"):
    # Warmup
    for _ in range(warmups):
        outputs = model(input_ids)
        if model.training:
            loss = outputs["logits"].mean()
            loss.backward()
            model.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    for _ in range(steps):
        outputs = model(input_ids)
        if model.training:
            loss = outputs["logits"].mean()
            loss.backward()
            model.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    avg_time = (end_time - start_time) / steps * 1000 # ms
    peak_mem = 0.0
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2) # MB

    batch_size, seq_len = input_ids.shape
    tokens_per_sec = (batch_size * seq_len) / (avg_time / 1000)

    params = sum(p.numel() for p in model.parameters()) / 1e6 # M

    mode = "Train" if model.training else "Eval"
    print(f"{label: <35} | {mode: <5} | {avg_time:8.2f} ms | {tokens_per_sec:10.0f} | {peak_mem:8.2f} MB | {params:6.1f}M")

    # Check for backward support
    if model.training:
        if "Triton" in label:
             print(f"  Note: {label} training currently falls back to PyTorch for backward pass.")

    return avg_time, peak_mem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--top_m", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    config = load_config_from_yaml(ModelConfig, args.model_config)

    print(f"{'Implementation': <35} | {'Mode': <5} | {'Latency': >8}    | {'Tok/s': >10} | {'Peak Mem': >8} | {'Params'}")
    print("-" * 100)

    for seq_len in args.seq_lens:
        print(f"\nSequence Length: {seq_len}")
        config.max_seq_len = seq_len
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=args.device)

        # SDPA Baseline
        config.attention_impl = "sdpa"
        model = DecoderForCausalLM(config).to(args.device)
        benchmark(model.eval(), input_ids, "SDPA Baseline", device=args.device)
        benchmark(model.train(), input_ids, "SDPA Baseline", device=args.device)

        for m in args.top_m:
            config.vayu_top_m_blocks = m

            # PyTorch Block
            config.attention_impl = "vayusphere_block"
            model = DecoderForCausalLM(config).to(args.device)
            benchmark(model.eval(), input_ids, f"VayuBlock PyTorch (m={m})", device=args.device)
            benchmark(model.train(), input_ids, f"VayuBlock PyTorch (m={m})", device=args.device)

            # Triton Block (if available)
            if args.device == "cuda":
                config.attention_impl = "vayusphere_block_triton_eval"
                config.vayu_pair_scorer = "cosine"
                model = DecoderForCausalLM(config).to(args.device).eval()
                benchmark(model, input_ids, f"VayuBlock Triton Cos (m={m})", device=args.device)

                config.vayu_pair_scorer = "linear"
                model = DecoderForCausalLM(config).to(args.device).eval()
                benchmark(model, input_ids, f"VayuBlock Triton Lin (m={m})", device=args.device)

if __name__ == "__main__":
    main()
