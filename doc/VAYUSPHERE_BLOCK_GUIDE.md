# VayuSphere Block Attention & 6GB KD Training Guide

This document explains how to run the new experimental VayuSphere-Fused Block Attention path and the memory-optimized Knowledge Distillation (KD) training.

## 1. VayuSphere Block Attention

The new attention path is gated by the `attention_impl` field in the model configuration.

### Configuration Options
- `attention_impl`:
  - `sdpa`: Standard PyTorch Flash Attention (Default).
  - `vayusphere_block`: VayuSphere Block-Sparse Attention (PyTorch path).
  - `vayusphere_block_triton_eval`: Triton-fused forward kernel (Eval only).
- `vayu_block_size`: Number of tokens per block (Default: 64).
- `vayu_top_m_blocks`: Number of key blocks to route to (Default: 2).
- `vayu_pair_scorer`: Scorer type (`cosine`, `linear`, `mlp`, `rbfkan`).
- `vayu_use_triton_eval`: Enable Triton during evaluation (Default: true).

### How to Run

#### Smoke Test (Sanity Check)
Verifies forward and backward passes, shape, and routing diagnostics.
```bash
python scripts/smoke_vayusphere_block.py \
  --model_config configs/model/vayusphere_block_nano_6gb.yaml \
  --device cuda
```

#### Performance Benchmark
Compares latency and memory of different implementations.
```bash
python scripts/benchmark_vayusphere_block.py \
  --model_config configs/model/vayusphere_block_nano_6gb.yaml \
  --seq_lens 512 1024 2048 \
  --device cuda
```

---

## 2. 6GB KD Training (Memory-Safe)

Optimized for training on low-VRAM hardware (e.g., 6GB Laptop GPUs).

### Key Features
- **CPU Teacher Support**: Offloads teacher model to CPU to save GPU memory.
- **Top-k Compression**: Only transfers top-k teacher logits (e.g., top 128) to GPU.
- **Proactive Cleanup**: Deletes large intermediate tensors immediately.

### Configuration (Training)
In `configs/train/laptop_6gb_block_kd.yaml`:
- `teacher_device: auto` (Detects if teacher fits on GPU, falls back to CPU if OOM).
- `topk_logits: 128` (Number of logits to keep for KD loss).

### How to Run

#### Tiny KD Smoke Test (Quick Verify)
Uses a synthetic dataset and small teacher to verify the KD loop.
```bash
python scripts/train_kd.py \
  --student_config configs/model/vayusphere_block_nano_6gb.yaml \
  --teacher_config distilgpt2 \
  --train_config configs/train/laptop_6gb_block_smoke.yaml \
  --teacher_device cpu \
  --topk_logits 10
```

#### Full KD Training
```bash
python scripts/train_kd.py \
  --model_config configs/model/vayusphere_block_nano_6gb.yaml \
  --teacher_config distilgpt2 \
  --train_config configs/train/laptop_6gb_block_kd.yaml \
  --data_dir data/fineweb_edu_gpt2_10m \
  --teacher_device auto
```

---

## 3. Triton Usage & Fallbacks

The Triton fused kernel is high-performance but has specific requirements:
- **Environment**: Linux + NVIDIA GPU.
- **Dtype**: Optimized for `fp16` and `bf16`.
- **Fallbacks**: The code automatically falls back to the PyTorch reference path if:
  - Triton is not installed.
  - Device is CPU.
  - Scorer is `mlp` or `rbfkan`.
  - Training mode is active (Triton kernel is forward-only in v0.1).
  - Sequence length is not divisible by block size.
