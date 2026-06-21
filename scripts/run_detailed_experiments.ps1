# Helper script to run the VayuSphere v0.2 validation studies

$choices = @(
    "1. Re-evaluate existing checkpoints (my_ablation_run)",
    "2. Run Scientific Control Test (frozen random vs trained centroids)",
    "3. Run 2x2 Confound Sweep (pre- vs post-RoPE x all vs top-k8)",
    "4. Run Alpha Sweep (alpha = [0.05, 0.10, 0.20, 0.40])",
    "5. Run Target Sweep (Q vs K vs QK targets)",
    "6. Run Multi-Seed Validation (seeds 42, 123, 777)",
    "7. Exit"
)

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "         VayuSphere v0.2 Detailed Experiment Suite            " -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
foreach ($choice in $choices) {
    Write-Host $choice
}
Write-Host "==============================================================" -ForegroundColor Cyan

$selection = Read-Host "Select an option [1-7]"

$env:PYTHONPATH="src"

switch ($selection) {
    "1" {
        Write-Host "Starting: Re-evaluating existing checkpoints on 100K tokens..." -ForegroundColor Yellow
        python scripts/reevaluate_checkpoints.py --run_dir outputs/ablate_v2/my_ablation_run --data_dir data/fineweb_edu_gpt2_10m --max_eval_batches 200
    }
    "2" {
        Write-Host "Starting: Scientific Control Test (10K steps)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode control_test --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --seed 42 --max_steps 10000 --run_name control_run
    }
    "3" {
        Write-Host "Starting: 2x2 Confound Sweep (10K steps)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode confound_sweep --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --seed 42 --max_steps 10000 --run_name confound_sweep
    }
    "4" {
        Write-Host "Starting: Alpha Sweep (10K steps)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode alpha_sweep --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --seed 42 --max_steps 10000 --run_name alpha_sweep
    }
    "5" {
        Write-Host "Starting: Target Sweep (10K steps)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode target_sweep --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --seed 42 --max_steps 10000 --run_name target_sweep
    }
    "6" {
        Write-Host "Starting: Multi-Seed Validation (10K steps for 3 seeds)..." -ForegroundColor Yellow
        python scripts/ablate_v2.py --mode multi_seed --train_config configs/train/laptop_6gb_10m.yaml --data_dir data/fineweb_edu_gpt2_10m --max_steps 10000 --run_name multi_seed_run
    }
    "7" {
        Write-Host "Exiting." -ForegroundColor Green
    }
    default {
        Write-Host "Invalid selection." -ForegroundColor Red
    }
}
