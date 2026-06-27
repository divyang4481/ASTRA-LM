# Build and Setup Guide for ASTRA-LM

This document provides instructions for setting up the environment and building ASTRA-LM with support for VayuSphere-Fused Block Attention.

## 1. Environment Setup

ASTRA-LM requires Python 3.10+ and a recent version of PyTorch (>=2.0.0).

### Standard Installation
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Triton Support (Optional, Linux NVIDIA GPU only)
Triton is required for high-performance block-sparse attention kernels.
```bash
pip install triton>=2.1.0
```

## 2. Verification

After installation, you can verify the setup by running the smoke tests:

### Forward Pass Smoke Test
```bash
python scripts/smoke_forward.py
```

### VayuSphere Block Attention Smoke Test
```bash
python scripts/smoke_vayusphere_block.py --model_config configs/model/vayusphere_block_nano_6gb.yaml
```

## 3. Configuration Changes for VayuSphere v0.1

To enable the new block attention path, modify your model configuration YAML:

```yaml
attention_impl: vayusphere_block  # Use "vayusphere_block_triton_eval" for Triton inference
vayu_block_size: 64
vayu_top_m_blocks: 2
vayu_pair_scorer: linear  # Options: cosine, linear, mlp, rbfkan
```

## 4. Building for 6GB KD Training

For laptop-class GPUs with 6GB VRAM, ensure you use the optimized training configs and teacher offloading:

```bash
# Example KD run command
python scripts/train_kd.py \
  --student_config configs/model/vayusphere_block_nano_6gb.yaml \
  --teacher_config distilgpt2 \
  --train_config configs/train/laptop_6gb_block_kd.yaml \
  --teacher_device auto
```

This configuration uses `topk_logits: 128` and `teacher_device: auto` to minimize memory pressure.
