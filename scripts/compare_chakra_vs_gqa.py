import argparse
import subprocess
import os
import json
import time
import sys

def run_training(model_config, train_config, data_dir, output_dir, device="cuda"):
    cmd = [
        sys.executable, "scripts/train.py",
        "--model_config", model_config,
        "--train_config", train_config,
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--device", device
    ]
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    if result.returncode != 0:
        print(f"Error running training for {model_config}:")
        print(result.stderr)
        return None, " ".join(cmd)

    # Try to extract metrics from stdout
    metrics = {
        "duration": end_time - start_time,
        "stdout": result.stdout,
        "command": " ".join(cmd)
    }

    # Simple extraction of last logged metrics
    lines = result.stdout.splitlines()
    for line in reversed(lines):
        if "Step" in line and "Loss:" in line:
            # Step 20000/20000 | Loss: 4.5678 | Candidate Ratio: 25.00% | LR: 3.00e-05 | Tok/s: 1234
            try:
                parts = line.split("|")
                for part in parts:
                    if "Loss:" in part:
                        metrics["final_train_loss"] = float(part.split(":")[1].strip().split()[0])
                    if "Candidate Ratio:" in part:
                        metrics["candidate_ratio"] = part.split(":")[1].strip()
                    if "Tok/s:" in part:
                        metrics["tokens_per_sec"] = float(part.split(":")[1].strip())
                    if "Mem:" in part:
                        metrics["max_cuda_memory"] = part.split(":")[1].strip()
                break
            except Exception:
                pass

        if "Eval Step" in line and "Loss:" in line:
            try:
                # Eval Step 20000 | Loss: 4.6789 | Perplexity: 107.65
                parts = line.split("|")
                for part in parts:
                    if "Loss:" in part:
                        metrics["eval_loss"] = float(part.split(":")[1].strip())
                # Don't break here, we want the LAST eval step
            except Exception:
                pass

    checkpoint_path = os.path.join(output_dir, "checkpoint-final.pt")
    # Actually trainer saves as checkpoint-{step}.pt, find the last one
    if os.path.exists(output_dir):
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-") and f.endswith(".pt")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            checkpoint_path = os.path.join(output_dir, latest_checkpoint)

    metrics["checkpoint_path"] = checkpoint_path

    return metrics, " ".join(cmd)

def main():
    parser = argparse.ArgumentParser(description="Compare ASTRA (CHAKRA) vs PRISM (GQA)")
    parser.add_argument("--astra_config", type=str, default="configs/model/astra_nano_6gb.yaml")
    parser.add_argument("--prism_config", type=str, default="configs/model/prism_gqa_baseline.yaml")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_base", type=str, default="outputs/comparison")

    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)

    report = {}

    # 1. Train ASTRA
    print("=== Training ASTRA (CHAKRA) ===")
    astra_output = os.path.join(args.output_base, "astra")
    astra_metrics, astra_cmd = run_training(args.astra_config, args.train_config, args.data_dir, astra_output, args.device)
    report["astra"] = astra_metrics

    # 2. Train PRISM
    print("\n=== Training PRISM (GQA) ===")
    prism_output = os.path.join(args.output_base, "prism")
    prism_metrics, prism_cmd = run_training(args.prism_config, args.train_config, args.data_dir, prism_output, args.device)
    report["prism"] = prism_metrics

    # Save report
    report_path = os.path.join(args.output_base, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nComparison complete! Report saved to {report_path}")

    if astra_metrics and prism_metrics:
        print("\nSummary:")
        print(f"{'Metric':<20} | {'ASTRA':<15} | {'PRISM':<15}")
        print("-" * 56)
        for key in ["final_train_loss", "eval_loss", "tokens_per_sec", "candidate_ratio"]:
            v1 = astra_metrics.get(key, "N/A")
            v2 = prism_metrics.get(key, "N/A")
            print(f"{key:<20} | {str(v1):<15} | {str(v2):<15}")

if __name__ == "__main__":
    main()
