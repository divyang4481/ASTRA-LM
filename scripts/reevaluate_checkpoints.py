import argparse
import logging
import os
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.train.checkpoint import load_checkpoint
from astra_lm.data.dataset import PretokenizedDataset
from astra_lm.data.collator import CausalLMCollator
from astra_lm.utils import load_config_from_yaml
from astra_lm.eval.perplexity import evaluate_perplexity


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved checkpoints on a larger validation set."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="outputs/ablate_v2/my_ablation_run",
        help="Path to the ablation run directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/fineweb_edu_gpt2_10m",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=100,
        help="Number of eval batches. Set to -1 for full validation.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to run on (cuda or cpu)."
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated list of variant names to evaluate. If None, auto-discovers all.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    # Determine validation data path
    val_path = os.path.join(args.data_dir, "val.npy")
    if not os.path.exists(val_path):
        val_path = os.path.join(args.data_dir, "validation.npy")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found in {args.data_dir}")

    # Base model config path
    base_model_config = "configs/model/gpt_nano_6gb.yaml"
    if not os.path.exists(base_model_config):
        raise FileNotFoundError(f"Base model config not found at {base_model_config}")

    # Discover immediate child directories
    if not os.path.exists(args.run_dir):
        raise FileNotFoundError(f"Run directory not found at {args.run_dir}")

    all_subdirs = [d for d in os.listdir(args.run_dir) if os.path.isdir(os.path.join(args.run_dir, d))]

    evaluation_targets = []
    # Check if we have nested seed folders (e.g., seed42, seed123)
    has_seed_folders = any(d.startswith("seed") for d in all_subdirs)

    if has_seed_folders:
        for sd in all_subdirs:
            if sd.startswith("seed"):
                seed_dir = os.path.join(args.run_dir, sd)
                variants_under_seed = [v for v in os.listdir(seed_dir) if os.path.isdir(os.path.join(seed_dir, v))]
                for v in variants_under_seed:
                    variant_dir = os.path.join(seed_dir, v)
                    checkpoints = [f for f in os.listdir(variant_dir) if f.startswith("checkpoint-") and f.endswith(".pt")]
                    if checkpoints:
                        evaluation_targets.append((f"{sd}/{v}", variant_dir))
    else:
        for sd in all_subdirs:
            variant_dir = os.path.join(args.run_dir, sd)
            checkpoints = [f for f in os.listdir(variant_dir) if f.startswith("checkpoint-") and f.endswith(".pt")]
            if checkpoints:
                evaluation_targets.append((sd, variant_dir))

    # Apply variants filter if provided
    if args.variants:
        filter_list = [v.strip() for v in args.variants.split(",")]
        filtered_targets = []
        for name, variant_dir in evaluation_targets:
            base_name = name.split("/")[-1]
            if name in filter_list or base_name in filter_list:
                filtered_targets.append((name, variant_dir))
        evaluation_targets = filtered_targets

    results = []

    # 100 batches is approx 100 * 512 = 51,200 tokens. Full validation is 500K tokens.
    max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

    # Load dataset once
    # We will temporarily load ModelConfig to get max_seq_len
    temp_m_cfg = load_config_from_yaml(ModelConfig, base_model_config)
    eval_dataset = PretokenizedDataset(val_path, seq_len=temp_m_cfg.max_seq_len)
    collator = CausalLMCollator()
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=1,  # Eval batch size is 1 as per config
        shuffle=False,
        collate_fn=collator,
    )

    logger.info(
        f"Re-evaluating checkpoints under {args.run_dir} using up to {max_eval_batches if max_eval_batches else 'ALL'} batches."
    )

    for name, variant_dir in evaluation_targets:
        # Find the latest checkpoint
        checkpoints = [
            f
            for f in os.listdir(variant_dir)
            if f.startswith("checkpoint-") and f.endswith(".pt")
        ]
        if not checkpoints:
            logger.warning(f"No checkpoints found in {variant_dir}. Skipping.")
            continue

        # Sort and pick the latest (or largest step number)
        def extract_step(f):
            try:
                return int(f.split("-")[-1].split(".")[0])
            except ValueError:
                return -1

        checkpoints.sort(key=extract_step)
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(variant_dir, latest_checkpoint)

        logger.info(f"Re-evaluating {name} from {checkpoint_path}...")

        # Load variant model config
        variant_cfg_path = os.path.join(variant_dir, "tmp_config.yaml")
        if os.path.exists(variant_cfg_path):
            m_cfg = load_config_from_yaml(ModelConfig, variant_cfg_path)
        else:
            # Fallback for older runs: parse config updates from the variant name
            m_cfg = load_config_from_yaml(ModelConfig, base_model_config)
            if "learned_temp" in name:
                m_cfg.use_learned_attention_temp = True
            elif "vs_" in name or "vayusphere" in name or "scale_" in name or "tangent_" in name:
                m_cfg.use_vayusphere = True
                if "tangent_scale" in name:
                    m_cfg.vayusphere_mode = "tangent_scale"
                    m_cfg.vayusphere_scale_alpha = 0.1
                elif "tangent" in name:
                    m_cfg.vayusphere_mode = "tangent"
                else:
                    m_cfg.vayusphere_mode = "scale"
                
                m_cfg.vayusphere_alpha = 0.1
                
                if "topk8" in name or "topk_8" in name:
                    m_cfg.vayusphere_topk_centroids = 8
                else:
                    m_cfg.vayusphere_topk_centroids = -1
                    
                if "pre_rope" in name or "prerope" in name:
                    m_cfg.vayusphere_apply_stage = "pre_rope"
                else:
                    m_cfg.vayusphere_apply_stage = "post_rope"
                    
                if "frozen" in name or "freeze" in name:
                    m_cfg.vayusphere_freeze_centroids = True

        # Disable VayuSphere diagnostics during evaluation to speed it up
        m_cfg.vayusphere_enable_heavy_diagnostics = False
        m_cfg.vayusphere_diagnostics_every_n_steps = 1000

        # Initialize model
        model = DecoderForCausalLM(m_cfg)

        # Load state dict
        try:
            load_checkpoint(checkpoint_path, model, map_location=device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            continue

        model.to(device)

        # Run evaluation
        eval_results = evaluate_perplexity(
            model=model,
            dataloader=eval_dl,
            device=torch.device(device),
            max_batches=max_eval_batches,
        )

        logger.info(
            f"Results for {name}: Loss = {eval_results['loss']:.4f}, Perplexity = {eval_results['perplexity']:.4f}"
        )

        results.append(
            {
                "name": name,
                "loss": eval_results["loss"],
                "perplexity": eval_results["perplexity"],
                "checkpoint": latest_checkpoint,
                "batches_evaluated": (
                    len(eval_dl)
                    if max_eval_batches is None
                    else min(len(eval_dl), max_eval_batches)
                ),
                "tokens_evaluated": (
                    len(eval_dl)
                    if max_eval_batches is None
                    else min(len(eval_dl), max_eval_batches)
                )
                * temp_m_cfg.max_seq_len,
            }
        )

        # Free GPU memory
        del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        logger.error("No variants evaluated.")
        return

    # Print Report
    df_results = pd.DataFrame(results)

    # Calculate Deltas vs Baseline
    delta_col = []
    for r in results:
        name = r["name"]
        loss = r["loss"]
        if "/" in name:
            seed_prefix = name.split("/")[0] + "/"
            baseline_loss = None
            for other in results:
                if other["name"] == f"{seed_prefix}A_baseline":
                    baseline_loss = other["loss"]
                    break
            if baseline_loss is not None:
                delta_col.append(loss - baseline_loss)
            else:
                delta_col.append(0.0)
        else:
            baseline_loss = None
            for other in results:
                if other["name"] == "A_baseline":
                    baseline_loss = other["loss"]
                    break
            if baseline_loss is not None:
                delta_col.append(loss - baseline_loss)
            else:
                delta_col.append(0.0)

    df_results["delta_vs_baseline"] = delta_col

    print("\n" + "=" * 60)
    print("CHECKPOINT RE-EVALUATION REPORT")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("=" * 60)

    # Save to CSV
    csv_path = os.path.join(args.run_dir, "reevaluation_results.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved re-evaluation report to {csv_path}")


if __name__ == "__main__":
    main()
