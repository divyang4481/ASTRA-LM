# ASTRA-LM: Adaptive Spherical Transformer for Reasoning Architecture

ASTRA-LM is a resource-efficient, low-VRAM decoder transformer architecture designed for running and training LLMs on consumer-grade hardware (such as 6 GB NVIDIA laptop GPUs) and free cloud accelerators (like Kaggle or Colab). 

It is built upon the **DHRUVA Transformer** decoder and introduces **CHAKRA Attention**, which reframes self-attention as a structured geometric search on a hypersphere to filter out irrelevant key tokens before computing attention weights.

---

## Architecture Overview

```
Input Tokens  -->  Embeddings + RoPE  -->  N × DHRUVA Blocks  -->  RMSNorm  -->  LM Head
```

Inside each **DHRUVA Block**, the forward flow is highly modular and config-gated:
1. **RMSNorm (Pre-Norm)**
2. **CHAKRA Attention** (or standard Grouped Query Attention control baseline)
   * **Local sliding window** (always included to preserve syntactic structure)
   * **Hyperspherical Routing** (projects queries and keys onto an $N$-dimensional sphere and groups them into angular buckets)
   * **Exact QK Softmax** computed only on selected candidate buckets
3. **AKASHA Memory Manager** (optional gated mixing of local attention with distant anchor memories)
4. **SURYA Mixer** (periodic FFT/DCT global sequence mixing to prevent sparse attention signal loss - *disabled by default*)
5. **INDRA Phase Layer** (lightweight real-valued phase/magnitude gate on hidden states - *disabled by default*)
6. **SwiGLU MLP / FOCK-FFN** (standard SwiGLU FFN or compact Chebyshev basis FFN - *disabled by default*)

---

## Project Structure

```
astra-lm/
├── configs/                  # YAML configurations
│   ├── model/
│   │   ├── astra_nano_6gb.yaml       # Targets 6GB GPU (~20M-60M params)
│   │   └── prism_gqa_baseline.yaml    # Control baseline model
│   └── train/
│       └── smoke.yaml                # Quick pretraining sanity check
├── src/astra_lm/             # Source package code
│   ├── data/
│   │   ├── collator.py               # Causal language modeling collator
│   │   └── dataset.py                # Synthetic and Pretokenized datasets
│   ├── distill/
│   │   ├── kd_losses.py              # KL and top-k logit distillation losses
│   │   └── teacher.py                # Teacher loading and parameter freezing
│   ├── eval/
│   │   └── perplexity.py             # Validation perplexity evaluator
│   ├── model/
│   │   ├── config.py                 # dataclass parsing configuration
│   │   ├── decoder.py                # Top-level Decoder Causal LM
│   │   ├── block.py                  # DHRUVA decoder block
│   │   ├── attention_gqa.py          # Baseline GQA implementation
│   │   ├── chakra_attention.py       # CHAKRA Routing attention
│   │   ├── sphere_bucket.py          # Spherical projection & bucketing
│   │   ├── akasha_memory.py          # AKASHA memory manager
│   │   ├── surya_mixer.py            # SURYA Spectral Mixer (Disabled)
│   │   ├── indra_phase.py            # INDRA Phase Gating (Disabled)
│   │   ├── fock_ffn.py               # FOCK Chebyshev FFN (Disabled)
│   │   ├── norms.py                  # RMSNorm module
│   │   ├── embeddings.py             # Token embeddings
│   │   └── rope.py                   # Rotary position embeddings
│   ├── train/
│   │   ├── trainer.py                # Pretraining trainer with CHAKRA logs
│   │   ├── kd_trainer.py             # Knowledge distillation trainer
│   │   ├── checkpoint.py             # Save, load, and prune checkpoints
│   │   └── optimizer.py              # AdamW weight decay separator and scheduler
│   └── utils/
│       └── config_utils.py           # YAML config parsing helpers
├── scripts/                  # Run scripts
│   ├── train.py                  # Single model pretraining entrypoint
│   ├── train_kd.py               # Distillation training entrypoint
│   ├── generate.py               # Logits sampling text generation
│   └── smoke_forward.py          # Forward shape and diagnostics verify
├── tests/                    # Comprehensive unit tests
│   ├── test_shapes.py            # Verify layer tensor dimensions
│   ├── test_causal_mask.py       # Ensure attention masks are strictly causal
│   ├── test_no_future_leakage.py # Validate causal gradients (zero leakage)
│   ├── test_chakra_candidate_mask.py  # Check sphere matching & neighboring
│   ├── test_local_window_always_included.py # Sanity check sliding window
│   ├── test_bucket_assignment_deterministic.py # Confirm scale-invariant projection
│   ├── test_one_training_step.py # Assert optimizer weights update
│   └── test_checkpoint_resume.py # Verify weights and logits loading matches
├── pyproject.toml            # PEP 621 package metadata
└── requirements.txt          # Portable dependencies file
```

---

## Installation & Setup

ASTRA-LM uses standard python package definitions, making it compatible with `uv`, standard `pip` or virtualenvs, and `conda` environments.

### CUDA Support Verification (Windows/Linux)
Before training, verify your PyTorch installation has CUDA support:
```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda build=', torch.version.cuda); print('cuda available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

If `cuda available` is `False`, you may need to reinstall PyTorch with the correct CUDA wheel index:
```powershell
# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### Using `uv` (Fastest)
If you have `uv` installed, setting up the environment takes seconds:
```powershell
uv sync
```

### Using standard `pip` + virtualenv
```powershell
# Create virtual environment named astra-lm-env
python -m venv astra-lm-env

# Activate virtual environment
astra-lm-env\Scripts\activate      # Windows PowerShell
source astra-lm-env/bin/activate    # Linux / WSL

# Install dependencies in editable mode
pip install -r requirements.txt
pip install -e .
```

### Using Conda
```powershell
# Create a conda environment named astra-lm-env
conda create -n astra-lm-env python=3.11 -y

# Activate the conda environment
conda activate astra-lm-env

# Install dependencies in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## Usage Guide

### 1. Verification (Smoke Test)
Run a quick, 2-layer forward pass to verify embedding shapes, logits, and CHAKRA candidate routing diagnostics on your system:
```powershell
$env:PYTHONPATH="src"
python scripts/smoke_forward.py
```

### 2. Pretraining Sanity Check (Smoke Training)
To make sure the training loop, checkpointing, and optimizers run without error. **Note:** `smoke.yaml` uses synthetic data by default.
```powershell
$env:PYTHONPATH="src"
python scripts/train.py --model_config configs/model/astra_nano_6gb.yaml --train_config configs/train/smoke.yaml
```

### 3. Serious Pretraining on Real Data
For real training, you must first prepare the data and then point the trainer to it using `--data_dir`.

#### Data Preparation (FineWeb-Edu)
```powershell
$env:PYTHONPATH="src"
python scripts/prepare_gpt2_pretrain_data.py `
  --dataset HuggingFaceFW/fineweb-edu `
  --name sample-10BT `
  --tokenizer gpt2 `
  --train_tokens 10000000 `
  --val_tokens 500000 `
  --out_dir data/fineweb_edu_gpt2_10m
```

#### Laptop 6GB Training (10M tokens)
```powershell
python scripts/train.py `
  --model_config configs/model/astra_nano_6gb.yaml `
  --train_config configs/train/laptop_6gb_10m.yaml `
  --data_dir data/fineweb_edu_gpt2_10m `
  --device cuda
```

### 4. CHAKRA vs GQA Comparison
Compare ASTRA (CHAKRA) against a PRISM (GQA) baseline on the same data:
```powershell
python scripts/compare_chakra_vs_gqa.py `
  --train_config configs/train/laptop_6gb_10m.yaml `
  --data_dir data/fineweb_edu_gpt2_10m
```

### 5. Text Generation
Generate text using a trained model checkpoint (runs a logits-sampling sequence builder):
```powershell
$env:PYTHONPATH="src"
python scripts/generate.py --checkpoint outputs/smoke/checkpoint-10.pt --model_config configs/model/astra_nano_6gb.yaml --prompt "Deep learning is"
```

### 6. Knowledge Distillation (DRONA-KD)
Train a student DHRUVA model under the guidance of a **trained** teacher model. A teacher checkpoint is required for serious KD.

```powershell
$env:PYTHONPATH="src"
python scripts/train_kd.py `
  --student_config configs/model/astra_nano_6gb.yaml `
  --teacher_config configs/model/prism_gqa_baseline.yaml `
  --teacher_checkpoint outputs/prism_gqa_fineweb_100m/checkpoint-50000.pt `
  --train_config configs/train/kaggle_kd_100m.yaml `
  --data_dir data/fineweb_edu_gpt2_100m
```

---

## Running in Kaggle

Because ASTRA-LM does not depend on custom C++/CUDA compile steps and uses native PyTorch masking, it can be run in a Kaggle notebook immediately.

### Setup on Kaggle:
1. Create a new Kaggle Notebook.
2. In the right panel, select **GPU P100** under Accelerator (Single P100 is recommended as multi-GPU is not yet implemented).
3. Clone the repo and install:
   ```bash
   !git clone https://github.com/divyang4481/ASTRA-LM.git
   %cd ASTRA-LM
   !pip install -e .
   ```
4. Prepare data and run training:
   ```bash
   !export PYTHONPATH=src && python scripts/prepare_gpt2_pretrain_data.py \
       --train_tokens 100000000 \
       --val_tokens 2000000 \
       --out_dir data/fineweb_edu_gpt2_100m

   !export PYTHONPATH=src && python scripts/train.py \
       --model_config configs/model/astra_nano_6gb.yaml \
       --train_config configs/train/kaggle_100m.yaml \
       --data_dir data/fineweb_edu_gpt2_100m \
       --device cuda
   ```

---

## Running Tests

Verify the complete mathematical and architectural suite:
```powershell
$env:PYTHONPATH="src"
pytest
```
Tests assert:
* **Shapes**: Correct sizes of GQA, CHAKRA, and embeddings.
* **Causal mask structure**: Attention masks are strictly lower-triangular.
* **No future leakage**: Gradients for future positions are exactly zero.
* **Chakra routing logic**: Neighbors and exact matches include/exclude correct keys.
* **Local window preservation**: Local window tokens are never pruned by sphere matching.
* **Scale-invariance**: Two inputs pointing in the same direction but with different magnitudes project to the same bucket.
* **Checkpoint Resume**: Loaded checkpoints yield identical logits.
