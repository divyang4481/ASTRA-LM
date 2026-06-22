# Helper script to run the VayuSphere v0.2 validation studies

$choices = @(
    "1. Re-evaluate existing checkpoints (custom run_dir)",
    "2. Run Scientific Control Test (pre-RoPE: frozen random vs trained centroids)",
    "3. Run Post-RoPE Scientific Control Test (control_test_postrope)",
    "4. Run 2x2 Confound Sweep (pre- vs post-RoPE x all vs top-k8)",
    "5. Run Alpha Sweep (alpha = [0.05, 0.10, 0.20, 0.40] - default: post-RoPE)",
    "6. Run Target Sweep (Q vs K vs QK targets - default: post-RoPE)",
    "7. Run Final VayuSphere v0.2 Proof Validation (proof_postrope_topk8 across 3 seeds)",
    "8. Run Multi-Seed Validation (seeds 42, 123, 777)",
    "9. Exit"
)

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "         VayuSphere v0.2 Detailed Experiment Suite            " -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
foreach ($choice in $choices) {
    Write-Host $choice
}
Write-Host "==============================================================" -ForegroundColor Cyan

$selection = Read-Host "Select an option [1-9]"

$env:PYTHONPATH="src"

switch ($selection) {
    "1" {
        $run_dir = Read-Host "Enter run directory [default: outputs/ablate_v2/confound_sweep]"
        if ([string]::IsNullOrWhiteSpace($run_dir)) { $run_dir = "outputs/ablate_v2/confound_sweep" }
        Write-Host "Starting: Re-evaluating checkpoints under $run_dir on 100K tokens..." -ForegroundColor Yellow
        python scripts/reevaluate_checkpoints.py --run_dir $run_dir --data_dir data/fineweb_edu_gpt2_10m --max_eval_batches 200
    }
    "2" {
        Write-Host "Starting: Scientific Control Test (pre-RoPE, 10K steps)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode control_test --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --seed 42 --max_steps 10000 --run_name control_run
    }
    "3" {
        Write-Host "Starting: Post-RoPE Scientific Control Test (control_test_postrope, 10K steps)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode control_test_postrope --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --seed 42 --max_steps 10000 --run_name control_postrope
    }
    "4" {
        Write-Host "Starting: 2x2 Confound Sweep (10K steps)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode confound_sweep --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --seed 42 --max_steps 10000 --run_name confound_sweep
    }
    "5" {
        $pipeline = Read-Host "Enter pipeline for sweep (postrope_topk8 or prerope_topk8) [default: postrope_topk8]"
        if ([string]::IsNullOrWhiteSpace($pipeline)) { $pipeline = "postrope_topk8" }
        Write-Host "Starting: Alpha Sweep ($pipeline, 10K steps)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode alpha_sweep --pipeline $pipeline --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --seed 42 --max_steps 10000 --run_name alpha_sweep
    }
    "6" {
        $pipeline = Read-Host "Enter pipeline for sweep (postrope_topk8 or prerope_topk8) [default: postrope_topk8]"
        if ([string]::IsNullOrWhiteSpace($pipeline)) { $pipeline = "postrope_topk8" }
        Write-Host "Starting: Target Sweep ($pipeline, 10K steps)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode target_sweep --pipeline $pipeline --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --seed 42 --max_steps 10000 --run_name target_sweep
    }
    "7" {
        Write-Host "Starting: Final Proof Validation (proof_postrope_topk8, 10K steps across 3 seeds)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode proof_postrope_topk8 --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --max_steps 10000 --run_name proof_postrope_topk8
    }
    "8" {
        Write-Host "Starting: Multi-Seed Validation (10K steps for 3 seeds)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode multi_seed --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --max_steps 10000 --run_name multi_seed_run
    }
    "9" {
        Write-Host "Exiting." -ForegroundColor Green
    }
    default {
        Write-Host "Invalid selection." -ForegroundColor Red
    }
}

