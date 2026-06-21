import argparse
import logging
import os
import datetime
import yaml
import pandas as pd
import numpy as np
import torch
try:
    from scripts.compare_gpt_vs_vayusphere import run_experiment, make_initial_state_dict
except ModuleNotFoundError:
    from compare_gpt_vs_vayusphere import run_experiment, make_initial_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["standard", "control_test", "confound_sweep", "alpha_sweep", "target_sweep", "multi_seed"],
        default="standard",
        help="Ablation experiment mode"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{timestamp}_seed{args.seed}_{args.mode}"

    run_dir = os.path.join("outputs", "ablate_v2", run_name)
    os.makedirs(run_dir, exist_ok=True)

    base_model_config = "configs/model/gpt_nano_6gb.yaml"

    def get_config(updates, variant_run_dir):
        with open(base_model_config, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg.update(updates)
        tmp_path = os.path.join(variant_run_dir, "tmp_config.yaml")
        os.makedirs(variant_run_dir, exist_ok=True)
        with open(tmp_path, 'w') as f:
            yaml.dump(cfg, f)
        return tmp_path

    # Define variants per mode
    if args.mode == "standard":
        variants = [
            ("A_baseline", {}),
            ("B_learned_temp", {"use_learned_attention_temp": True}),
            ("C_vs_scale_v0.1", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_apply_stage": "post_rope"
            }),
            ("D_vs_scale_topk8", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
            ("E_vs_tangent_pre_rope_topk8", {
                "use_vayusphere": True,
                "vayusphere_mode": "tangent",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
        ]
    elif args.mode == "control_test":
        variants = [
            ("A_baseline", {}),
            ("B_learned_temp", {"use_learned_attention_temp": True}),
            ("D_vs_scale_topk8", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
            ("D_frozen_random_centroids_topk8_prerope", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope",
                "vayusphere_freeze_centroids": True
            }),
        ]
    elif args.mode == "confound_sweep":
        variants = [
            ("scale_all_postrope", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_apply_stage": "post_rope"
            }),
            ("scale_topk8_postrope", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "post_rope"
            }),
            ("scale_all_prerope", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_apply_stage": "pre_rope"
            }),
            ("scale_topk8_prerope", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
        ]
    elif args.mode == "alpha_sweep":
        variants = [
            ("alpha_0.05", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.05,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
            ("alpha_0.10", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
            ("alpha_0.20", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.2,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
            ("alpha_0.40", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.4,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
        ]
    elif args.mode == "target_sweep":
        variants = [
            ("target_q", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope",
                "vayusphere_target": "q"
            }),
            ("target_k", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope",
                "vayusphere_target": "k"
            }),
            ("target_qk", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope",
                "vayusphere_target": "qk"
            }),
        ]
    elif args.mode == "multi_seed":
        variants = [
            ("A_baseline", {}),
            ("B_learned_temp", {"use_learned_attention_temp": True}),
            ("D_vs_scale_topk8", {
                "use_vayusphere": True,
                "vayusphere_mode": "scale",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
            ("E_vs_tangent_pre_rope_topk8", {
                "use_vayusphere": True,
                "vayusphere_mode": "tangent",
                "vayusphere_alpha": 0.1,
                "vayusphere_topk_centroids": 8,
                "vayusphere_apply_stage": "pre_rope"
            }),
        ]

    results = []

    if args.mode == "multi_seed":
        seeds = [42, 123, 777]
        for seed in seeds:
            # Generate seed-specific initial state dict
            initial_state = make_initial_state_dict(base_model_config, seed)
            
            for name, updates in variants:
                variant_run_dir = os.path.join(run_dir, f"seed{seed}", name)
                cfg_path = get_config(updates, variant_run_dir)
                
                res = run_experiment(
                    name=name,
                    model_config_path=cfg_path,
                    train_config_path=args.train_config,
                    data_dir=args.data_dir,
                    output_dir=variant_run_dir,
                    base_state_dict=initial_state,
                    seed=seed,
                    max_steps_override=args.max_steps
                )
                res.pop("state_dict")
                res["seed"] = seed
                results.append(res)
                
            del initial_state
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # Single-seed execution (standard, control_test, sweeps)
        initial_state = make_initial_state_dict(base_model_config, args.seed)
        
        for name, updates in variants:
            variant_run_dir = os.path.join(run_dir, name)
            cfg_path = get_config(updates, variant_run_dir)
            
            res = run_experiment(
                name=name,
                model_config_path=cfg_path,
                train_config_path=args.train_config,
                data_dir=args.data_dir,
                output_dir=variant_run_dir,
                base_state_dict=initial_state,
                seed=args.seed,
                max_steps_override=args.max_steps
            )
            res.pop("state_dict")
            results.append(res)

    df_results = pd.DataFrame(results)

    # Compute slowdown_vs_baseline
    if args.mode == "multi_seed":
        # Group by seed and calculate slowdown
        slowdowns = []
        for seed in [42, 123, 777]:
            seed_df = df_results[df_results["seed"] == seed]
            baseline_row = seed_df[seed_df["name"] == "A_baseline"]
            if not baseline_row.empty:
                baseline_tps = baseline_row.iloc[0]["tokens_per_sec"]
                for idx, row in seed_df.iterrows():
                    curr_tps = row["tokens_per_sec"]
                    slowdown = (baseline_tps / curr_tps) if curr_tps > 0 else 1.0
                    df_results.at[idx, "slowdown_vs_baseline"] = slowdown
            else:
                for idx in seed_df.index:
                    df_results.at[idx, "slowdown_vs_baseline"] = 1.0
    else:
        baseline_row = df_results[df_results["name"] == "A_baseline"]
        if baseline_row.empty and not df_results.empty:
            # If no A_baseline is present in sweep modes, use first variant as reference
            baseline_tps = df_results.iloc[0]["tokens_per_sec"]
        else:
            baseline_tps = baseline_row.iloc[0]["tokens_per_sec"] if not baseline_row.empty else 0.0

        df_results["slowdown_vs_baseline"] = df_results["tokens_per_sec"].apply(
            lambda x: (baseline_tps / x) if x > 0 and baseline_tps > 0 else 1.0
        )

    # Save details
    df_results.to_csv(os.path.join(run_dir, "ablation_results.csv"), index=False)

    # Display final reports
    print("\n" + "=" * 60)
    print(f"VAYUSPHERE V0.2 ABLATION REPORT ({args.mode.upper()} MODE)")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("=" * 60)

    # Multi-seed Aggregation
    if args.mode == "multi_seed":
        # Compute mean, std, mean delta, win count
        agg_rows = []
        
        # Calculate deltas for each seed run
        for idx, row in df_results.iterrows():
            seed = row["seed"]
            name = row["name"]
            baseline_row = df_results[(df_results["seed"] == seed) & (df_results["name"] == "A_baseline")]
            if not baseline_row.empty:
                df_results.at[idx, "delta_vs_baseline"] = row["loss"] - baseline_row.iloc[0]["loss"]
            else:
                df_results.at[idx, "delta_vs_baseline"] = 0.0

        for name in [v[0] for v in variants]:
            variant_df = df_results[df_results["name"] == name]
            
            mean_loss = variant_df["loss"].mean()
            std_loss = variant_df["loss"].std()
            mean_ppl = variant_df["perplexity"].mean()
            mean_tps = variant_df["tokens_per_sec"].mean()
            mean_mem = variant_df["peak_mem_mb"].mean()
            mean_slowdown = variant_df["slowdown_vs_baseline"].mean()
            mean_delta = variant_df["delta_vs_baseline"].mean()
            
            # Count wins (where loss is strictly lower than baseline for that seed)
            wins = 0
            for seed in [42, 123, 777]:
                v_row = variant_df[variant_df["seed"] == seed]
                b_row = df_results[(df_results["seed"] == seed) & (df_results["name"] == "A_baseline")]
                if not v_row.empty and not b_row.empty:
                    if v_row.iloc[0]["loss"] < b_row.iloc[0]["loss"]:
                        wins += 1

            agg_rows.append({
                "name": name,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "mean_delta_vs_baseline": mean_delta,
                "win_count": wins,
                "mean_perplexity": mean_ppl,
                "mean_tokens_per_sec": mean_tps,
                "mean_slowdown_vs_baseline": mean_slowdown,
                "mean_peak_mem_mb": mean_mem,
            })

        df_agg = pd.DataFrame(agg_rows)
        df_agg.to_csv(os.path.join(run_dir, "aggregate_mean_std.csv"), index=False)
        print("\n" + "=" * 60)
        print("AGGREGATED MULTI-SEED PERFORMANCE REPORT")
        print("=" * 60)
        print(df_agg.to_string(index=False))
        print("=" * 60)

    # Common step comparison logic (only for single seed modes where A_baseline is present)
    elif "A_baseline" in [v[0] for v in variants]:
        metrics_dfs = {}
        for name, _ in variants:
            m_path = os.path.join(run_dir, name, "metrics.csv")
            if os.path.exists(m_path):
                m_df = pd.read_csv(m_path)
                metrics_dfs[name] = m_df[m_df["eval_loss"].notna()]

        if len(metrics_dfs) >= 2 and "A_baseline" in metrics_dfs:
            common_steps = None
            for name, m_df in metrics_dfs.items():
                steps = set(m_df["step"].tolist())
                if common_steps is None: common_steps = steps
                else: common_steps = common_steps.intersection(steps)

            if common_steps:
                last_common_step = sorted(list(common_steps))[-1]
                baseline_val = metrics_dfs["A_baseline"][metrics_dfs["A_baseline"]["step"] == last_common_step]["eval_loss"].values[0]

                print(f"\nCOMMON STEP COMPARISON (Step {last_common_step})")
                print("-" * 60)
                print(f"{'Variant':<35} | {'Loss':<10} | {'Delta':<10} | {'Best':<10}")
                for name, _ in variants:
                    if name in metrics_dfs:
                        m_df = metrics_dfs[name]
                        current_val = m_df[m_df["step"] == last_common_step]["eval_loss"].values[0]
                        best_val = m_df[m_df["step"] <= last_common_step]["eval_loss"].min()
                        delta = current_val - baseline_val
                        print(f"{name:<35} | {current_val:.4f}     | {delta:+.4f}    | {best_val:.4f}")
                print("-" * 60)

if __name__ == "__main__":
    main()
