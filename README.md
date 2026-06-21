# ASTRA-LM: Adaptive Spherical Transformer for Reasoning Architecture

ASTRA-LM is a resource-efficient, low-VRAM decoder transformer architecture designed for running and training LLMs on consumer-grade hardware (such as 6 GB NVIDIA laptop GPUs) and free cloud accelerators (like Kaggle or Colab).

It is built upon a standard GPT-style baseline (RoPE, RMSNorm, SwiGLU, GQA) and introduces **VayuSphere**, a lightweight angular adapter that uses hyperspherical centroids to modulate query and key directions before attention.

---

## Architecture Overview

```
Input Tokens  -->  Embeddings + RoPE  -->  N × DHRUVA Blocks  -->  RMSNorm  -->  LM Head
```

Inside each **DHRUVA Block**, the forward flow is highly modular and config-gated:

1. **RMSNorm (Pre-Norm)**
2. **Standard Attention (SDPA)**
   - Uses PyTorch `scaled_dot_product_attention` for maximum performance (Flash Attention).
   - Supports Multi-Head Attention (MHA) or Grouped-Query Attention (GQA).
   - **VayuSphere Adapter** (Optional): A lightweight angular gate that projects Q/K onto a hypersphere and applies a learnable residual scale based on similarity to learned centroids.
3. **CHAKRA Attention** (Legacy Research Option)
   - **Local sliding window** (always included to preserve syntactic structure)
   - **Hyperspherical Routing** (projects queries and keys onto an $N$-dimensional sphere and groups them into angular buckets)
   - **Exact QK Softmax** computed only on selected candidate buckets
4. **AKASHA Memory Manager** (optional gated mixing of local attention with distant anchor memories)
5. **SURYA Mixer** (periodic FFT/DCT global sequence mixing to prevent sparse attention signal loss - _disabled by default_)
6. **INDRA Phase Layer** (lightweight real-valued phase/magnitude gate on hidden states - _disabled by default_)
7. **SwiGLU MLP / FOCK-FFN** (standard SwiGLU FFN or compact Chebyshev basis FFN - _disabled by default_)

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
│   │   ├── attention_sdpa.py         # Production SDPA/Flash Attention path
│   │   ├── vayusphere_adapter.py     # Lightweight angular Q/K adapter
│   │   ├── attention_gqa.py          # Baseline GQA implementation (Legacy)
│   │   ├── chakra_attention.py       # CHAKRA Routing attention (Legacy)
│   │   ├── sphere_bucket.py          # Spherical projection & bucketing (Legacy)
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
├── notebooks/                # Jupyter notebook guides (Colab/Kaggle)
│   └── astra_lm_cloud_training.ipynb
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
python scripts/train.py --model_config configs/model/gpt_nano_6gb.yaml --train_config configs/train/smoke.yaml
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
# Standard GPT Baseline
python scripts/train.py `
  --model_config configs/model/gpt_nano_6gb.yaml `
  --train_config configs/train/laptop_6gb_10m.yaml `
  --data_dir data/fineweb_edu_gpt2_10m `
  --device cuda

# VayuSphere GPT
python scripts/train.py `
  --model_config configs/model/vayusphere_gpt_nano_6gb.yaml `
  --train_config configs/train/laptop_6gb_10m.yaml `
  --data_dir data/fineweb_edu_gpt2_10m `
  --device cuda
```

### 4. Baseline vs VayuSphere Comparison

Fairly compare standard GPT against the VayuSphere-enabled model in one of two modes:

#### Mode A: Fair Scratch Comparison (`scratch_fair`)

Trains both GPT and VayuSphere from the exact same random starting weights:

```powershell
$env:PYTHONPATH="src"
python scripts/compare_gpt_vs_vayusphere.py `
  --train_config configs/train/laptop_6gb_10m.yaml `
  --data_dir data/fineweb_edu_gpt2_10m `
  --seed 42 `
  --mode scratch_fair
```

#### Mode B: Warm-Start / Retrofit Comparison (`warm_start`)

Trains a GPT baseline for a set number of steps, saves the checkpoint, and branches into continuing GPT baseline training vs converting it to VayuSphere and training under the same conditions:

```powershell
$env:PYTHONPATH="src"
python scripts/compare_gpt_vs_vayusphere.py `
  --train_config configs/train/laptop_6gb_10m.yaml `
  --data_dir data/fineweb_edu_gpt2_10m `
  --seed 42 `
  --mode warm_start `
  --warm_start_steps 10000
```

The resulting `comparison_results.csv` and detailed metric subfolders are saved in the timestamped run directory under `outputs/compare_gpt_vs_vayusphere/<timestamp>_seed<seed>_<mode>/`.

### 5. Ablation Study & Verification (VayuSphere v0.2)

VayuSphere v0.2 supports advanced ablation modes to systematically isolate architecture gains, test random centroid controls, resolve confounds, and measure throughput slowdowns.

#### Step 1: Re-evaluate Existing Checkpoints (Priority 1)
To verify if loss deltas are statistically sound (ruling out evaluation subset noise), re-evaluate saved checkpoints on a 10x larger validation set (e.g. 100K to 500K tokens):
```powershell
$env:PYTHONPATH="src"
python scripts/reevaluate_checkpoints.py `
  --run_dir outputs/ablate_v2/my_ablation_run `
  --data_dir data/fineweb_edu_gpt2_10m `
  --max_eval_batches 200
```
This produces `reevaluation_results.csv` within the run folder containing more precise evaluations of loss and perplexity.

#### Step 2: Running Ablation Sweeps and Control Runs
Execute a specific ablation mode using `scripts/ablate_v2.py`:
```powershell
$env:PYTHONPATH="src"
python scripts/ablate_v2.py `
  --mode control_test `
  --train_config configs/train/laptop_6gb_10m.yaml `
  --data_dir data/fineweb_edu_gpt2_10m `
  --seed 42 `
  --max_steps 10000 `
  --run_name my_control_run
```

Available ablation modes (`--mode`):

- **`control_test`**: Runs the critical scientific control variant `D_frozen_random_centroids_topk8_prerope` alongside the baseline, learned temperature, and trained VayuSphere D variant. (Centroids are kept frozen and initialized randomly to test if gains are due to content-dependent perturbations/regularization rather than learned semantic routing).
- **`confound_sweep`**: A clean 2x2 grid checking pre- vs. post-RoPE stages against all-centroids vs. top-k8 centroids to isolate which factor drives the performance gain.
- **`alpha_sweep`**: Sweeps `alpha = [0.05, 0.10, 0.20, 0.40]` on the D pipeline to inspect gate scaling limits.
- **`target_sweep`**: Compares targeting `q` vs. `k` vs. `qk` to see if scaling both compounds noise.
- **`multi_seed`**: Performs paired training runs across seeds `[42, 123, 777]` for baseline, learned temp, D, and E variants. Aggregates results into `aggregate_mean_std.csv` detailing win counts and mean/std metrics.
- **`standard`**: Runs baseline, learned temp, C, D, and E variants (pauses F tangent+scale).

You can load and visualize the results (including aggregated multi-seed tables and re-evaluation curves) using the updated `notebooks/analyze_experiments.ipynb` notebook.

#### Running via the Interactive Script
To make running these validation experiments easier, you can use the interactive helper script:
```powershell
./scripts/run_detailed_experiments.ps1
```
This script will prompt you with a menu to run any of the validation experiments or checkpoint re-evaluations with the correct environment variables set automatically.


### 6. Text Generation

Generate text using a trained model checkpoint (runs a logits-sampling sequence builder):

```powershell
$env:PYTHONPATH="src"
python scripts/generate.py --checkpoint outputs/smoke/checkpoint-10.pt --model_config configs/model/astra_nano_6gb.yaml --prompt "Deep learning is"
```

### 7. Knowledge Distillation (DRONA-KD)

Train a student DHRUVA model under the guidance of a teacher model. You can load a local model config and checkpoint, or dynamically fetch a pretrained teacher directly from Hugging Face (such as `gpt2-medium` or `gpt2`):

#### Laptop KD Training (using Hugging Face Teacher):

```powershell
$env:PYTHONPATH="src"
python scripts/train_kd.py `
  --student_config configs/model/vayusphere_gpt_nano_6gb.yaml `
  --teacher_config gpt2-medium `
  --train_config configs/train/laptop_distill.yaml `
  --data_dir data/fineweb_edu_gpt2_10m `
  --alpha 0.5 `
  --temperature 2.0 `
  --teacher_dtype 8bit
```

#### Dynamic Context Capping & Low VRAM Long Contexts:

- **Context Capping in KD**: Hugging Face teacher models have fixed maximum position embeddings (e.g., `1024` tokens for `gpt2-medium`). The distillation script dynamically queries the teacher config and caps the training sequence length to `min(student_max_seq_len, teacher_max_seq_len)` to avoid out-of-bound indexing crashes.
- **Low VRAM Long-Context Pretraining**: When pretraining a student model _directly_ (without the teacher model in memory, using `scripts/train.py`), you can train with context windows as large as **`8192`** on a 6 GB VRAM laptop. This is enabled by **CHAKRA Attention Routing**, which compresses key-value cache length by over **90%** (routing query tokens only to a tiny fraction of total key-value buckets), scaling memory consumption near-linearly instead of quadratically with sequence length.

---

## Running in Google Colab & Kaggle

Because ASTRA-LM does not depend on custom C++/CUDA compile steps and uses native PyTorch masking, it runs on Google Colab and Kaggle immediately.

We have created a ready-to-run Jupyter notebook: [astra_lm_cloud_training.ipynb](file:///c:/workspace/AI/ASTRA-LM/notebooks/astra_lm_cloud_training.ipynb).

### Setup and Upload Guide:

1. **Push your code to GitHub**: Since you are pair-programming in a Git repository, commit and push your latest changes (`git add .`, `git commit -m "Updates"`, `git push`).
2. **Open Google Colab or Kaggle**:
   - **Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com/) -> Select **Upload** -> Drag and drop the `notebooks/astra_lm_cloud_training.ipynb` file from your local computer.
   - **Kaggle**: Go to [kaggle.com/code](https://www.kaggle.com/code) -> Click **New Notebook** -> Select **File** -> **Import Notebook** -> Upload `notebooks/astra_lm_cloud_training.ipynb`.
3. **Select GPU Accelerator**:
   - **Colab**: Click **Runtime** -> **Change runtime type** -> Select **T4 GPU** or higher.
   - **Kaggle**: In the right sidebar panel, under **Accelerator**, select **GPU T4 x2** or **GPU P100**.
4. **Execute**: Run the notebook cells sequentially. The first cell will clone your repo and install the package dependencies:
   ```bash
   !git clone https://github.com/divyang4481/ASTRA-LM.git
   %cd ASTRA-LM
   !pip install -e .
   !pip install bitsandbytes accelerate
   ```

---

## Running Tests

Verify the complete mathematical and architectural suite:

```powershell
$env:PYTHONPATH="src"
pytest
```

Tests assert:

- **Shapes**: Correct sizes of GQA, CHAKRA, and embeddings.
- **Causal mask structure**: Attention masks are strictly lower-triangular.
- **No future leakage**: Gradients for future positions are exactly zero.
- **Chakra routing logic**: Neighbors and exact matches include/exclude correct keys.
- **Local window preservation**: Local window tokens are never pruned by sphere matching.
- **Scale-invariance**: Two inputs pointing in the same direction but with different magnitudes project to the same bucket.
- **Checkpoint Resume**: Loaded checkpoints yield identical logits.
