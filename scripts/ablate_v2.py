import argparse
import logging
import os
import datetime
import yaml
import pandas as pd
import numpy as np
import torch

try:
    from scripts.compare_gpt_vs_vayusphere import (
        run_experiment,
        make_initial_state_dict,
    )
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
        choices=[
            "standard",
            "control_test",
            "control_test_postrope",
            "confound_sweep",
            "alpha_sweep",
            "target_sweep",
            "multi_seed",
            "proof_postrope_topk8",
        ],
        default="standard",
        help="Ablation experiment mode",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["postrope_topk8", "prerope_topk8"],
        default="postrope_topk8",
        help="VayuSphere pipeline configuration to use for sweeps.",
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
        with open(base_model_config, "r") as f:
            cfg = yaml.safe_load(f)
        cfg.update(updates)
        tmp_path = os.path.join(variant_run_dir, "tmp_config.yaml")
        os.makedirs(variant_run_dir, exist_ok=True)
        with open(tmp_path, "w") as f:
            yaml.dump(cfg, f)
        return tmp_path

    # Determine the pipeline stage and topk centroids for sweeps
    if args.pipeline == "postrope_topk8":
        sweep_stage = "post_rope"
        sweep_topk = 8
    else:
        sweep_stage = "pre_rope"
        sweep_topk = 8

    # Define variants per mode
    if args.mode == "standard":
        variants = [
            ("A_baseline", {}),
            ("B_learned_temp", {"use_learned_attention_temp": True}),
            (
                "C_vs_scale_v0.1",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_apply_stage": "post_rope",
                },
            ),
            (
                "D_vs_scale_topk8",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "pre_rope",
                },
            ),
            (
                "E_vs_tangent_pre_rope_topk8",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "tangent",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "pre_rope",
                },
            ),
        ]
    elif args.mode == "control_test":
        variants = [
            ("A_baseline", {}),
            ("B_learned_temp", {"use_learned_attention_temp": True}),
            (
                "D_vs_scale_topk8",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "pre_rope",
                },
            ),
            (
                "D_frozen_random_centroids_topk8_prerope",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "pre_rope",
                    "vayusphere_freeze_centroids": True,
                },
            ),
        ]
    elif args.mode == "control_test_postrope":
        variants = [
            ("A_baseline", {}),
            ("B_learned_temp", {"use_learned_attention_temp": True}),
            (
                "scale_topk8_postrope",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "post_rope",
                },
            ),
            (
                "scale_topk8_postrope_frozen_random",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "post_rope",
                    "vayusphere_freeze_centroids": True,
                },
            ),
        ]
    elif args.mode == "confound_sweep":
        variants = [
            ("A_baseline", {}),
            ("B_learned_temp", {"use_learned_attention_temp": True}),
            (
                "scale_all_postrope",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_apply_stage": "post_rope",
                },
            ),
            (
                "scale_topk8_postrope",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "post_rope",
                },
            ),
            (
                "scale_all_prerope",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_apply_stage": "pre_rope",
                },
            ),
            (
                "scale_topk8_prerope",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "pre_rope",
                },
            ),
        ]
    elif args.mode == "alpha_sweep":
        variants = [
            (
                f"alpha_0.05_{args.pipeline}",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.05,
                    "vayusphere_topk_centroids": sweep_topk,
                    "vayusphere_apply_stage": sweep_stage,
                },
            ),
            (
                f"alpha_0.10_{args.pipeline}",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": sweep_topk,
                    "vayusphere_apply_stage": sweep_stage,
                },
            ),
            (
                f"alpha_0.20_{args.pipeline}",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.2,
                    "vayusphere_topk_centroids": sweep_topk,
                    "vayusphere_apply_stage": sweep_stage,
                },
            ),
            (
                f"alpha_0.40_{args.pipeline}",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.4,
                    "vayusphere_topk_centroids": sweep_topk,
                    "vayusphere_apply_stage": sweep_stage,
                },
            ),
        ]
    elif args.mode == "target_sweep":
        variants = [
            (
                f"target_q_{args.pipeline}",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": sweep_topk,
                    "vayusphere_apply_stage": sweep_stage,
                    "vayusphere_target": "q",
                },
            ),
            (
                f"target_k_{args.pipeline}",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": sweep_topk,
                    "vayusphere_apply_stage": sweep_stage,
                    "vayusphere_target": "k",
                },
            ),
            (
                f"target_qk_{args.pipeline}",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": sweep_topk,
                    "vayusphere_apply_stage": sweep_stage,
                    "vayusphere_target": "qk",
                },
            ),
        ]
    elif args.mode in ["multi_seed", "proof_postrope_topk8"]:
        variants = [
            ("A_baseline", {}),
            ("B_learned_temp", {"use_learned_attention_temp": True}),
            (
                "scale_topk8_postrope",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "post_rope",
                },
            ),
            (
                "scale_topk8_postrope_frozen_random",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_topk_centroids": 8,
                    "vayusphere_apply_stage": "post_rope",
                    "vayusphere_freeze_centroids": True,
                },
            ),
            (
                "scale_all_postrope",
                {
                    "use_vayusphere": True,
                    "vayusphere_mode": "scale",
                    "vayusphere_alpha": 0.1,
                    "vayusphere_apply_stage": "post_rope",
                },
            ),
        ]

    results = []

    if args.mode in ["multi_seed", "proof_postrope_topk8"]:
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
                    max_steps_override=args.max_steps,
                )
                res.pop("state_dict")
                res["seed"] = seed
                # Add summary fields
                res["apply_stage"] = updates.get("vayusphere_apply_stage", "post_rope" if updates.get("use_vayusphere", False) else "n/a")
                res["topk_centroids"] = updates.get("vayusphere_topk_centroids", -1 if updates.get("use_vayusphere", False) else "n/a")
                res["vayusphere_mode"] = updates.get("vayusphere_mode", "scale" if updates.get("use_vayusphere", False) else "n/a")
                res["vayusphere_alpha"] = updates.get("vayusphere_alpha", 0.1 if updates.get("use_vayusphere", False) else 0.0)
                res["vayusphere_target"] = updates.get("vayusphere_target", "qk" if updates.get("use_vayusphere", False) else "n/a")
                res["freeze_centroids"] = updates.get("vayusphere_freeze_centroids", False if updates.get("use_vayusphere", False) else "n/a")
                results.append(res)

            del initial_state
            import gc

            gc.collect()
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
                max_steps_override=args.max_steps,
            )
            res.pop("state_dict")
            # Add summary fields
            res["apply_stage"] = updates.get("vayusphere_apply_stage", "post_rope" if updates.get("use_vayusphere", False) else "n/a")
            res["topk_centroids"] = updates.get("vayusphere_topk_centroids", -1 if updates.get("use_vayusphere", False) else "n/a")
            res["vayusphere_mode"] = updates.get("vayusphere_mode", "scale" if updates.get("use_vayusphere", False) else "n/a")
            res["vayusphere_alpha"] = updates.get("vayusphere_alpha", 0.1 if updates.get("use_vayusphere", False) else 0.0)
            res["vayusphere_target"] = updates.get("vayusphere_target", "qk" if updates.get("use_vayusphere", False) else "n/a")
            res["freeze_centroids"] = updates.get("vayusphere_freeze_centroids", False if updates.get("use_vayusphere", False) else "n/a")
            results.append(res)

    df_results = pd.DataFrame(results)

    # Compute slowdown_vs_baseline / slowdown_vs_reference
    has_baseline = "A_baseline" in df_results["name"].values
    slowdown_col = "slowdown_vs_baseline" if has_baseline else "slowdown_vs_reference"

    if args.mode == "multi_seed":
        # Group by seed and calculate slowdown
        for seed in [42, 123, 777]:
            seed_df = df_results[df_results["seed"] == seed]
            baseline_row_seed = seed_df[seed_df["name"] == "A_baseline"]
            if not baseline_row_seed.empty:
                baseline_tps = baseline_row_seed.iloc[0]["tokens_per_sec"]
                for idx, row in seed_df.iterrows():
                    curr_tps = row["tokens_per_sec"]
                    slowdown = (baseline_tps / curr_tps) if curr_tps > 0 else 1.0
                    df_results.at[idx, slowdown_col] = slowdown
            else:
                first_tps = seed_df.iloc[0]["tokens_per_sec"] if not seed_df.empty else 1.0
                for idx, row in seed_df.iterrows():
                    curr_tps = row["tokens_per_sec"]
                    slowdown = (first_tps / curr_tps) if curr_tps > 0 else 1.0
                    df_results.at[idx, slowdown_col] = slowdown
    else:
        baseline_row = df_results[df_results["name"] == "A_baseline"]
        if baseline_row.empty and not df_results.empty:
            baseline_tps = df_results.iloc[0]["tokens_per_sec"]
        else:
            baseline_tps = (
                baseline_row.iloc[0]["tokens_per_sec"]
                if not baseline_row.empty
                else 0.0
            )

        df_results[slowdown_col] = df_results["tokens_per_sec"].apply(
            lambda x: (baseline_tps / x) if x > 0 and baseline_tps > 0 else 1.0
        )

    # Compute delta_vs_baseline for single seed runs
    if args.mode != "multi_seed":
        baseline_row = df_results[df_results["name"] == "A_baseline"]
        if not baseline_row.empty:
            baseline_loss = baseline_row.iloc[0]["loss"]
            df_results["delta_vs_baseline"] = df_results["loss"] - baseline_loss
        else:
            df_results["delta_vs_baseline"] = 0.0

    # Save details
    df_results.to_csv(os.path.join(run_dir, "ablation_results.csv"), index=False)

    # Display final reports
    print("\n" + "=" * 60)
    print(f"VAYUSPHERE V0.2 ABLATION REPORT ({args.mode.upper()} MODE)")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("=" * 60)

    # Detailed Gain Analysis (Single Seed)
    if args.mode != "multi_seed":
        baseline_loss = None
        temp_loss = None
        trained_loss = None
        frozen_loss = None
        trained_name = None
        
        for idx, row in df_results.iterrows():
            name = row["name"]
            loss = row["loss"]
            if name == "A_baseline":
                baseline_loss = loss
            elif name == "B_learned_temp":
                temp_loss = loss
            elif name in ["scale_topk8_postrope", "D_vs_scale_topk8"]:
                trained_loss = loss
                trained_name = name
            elif name in ["scale_topk8_postrope_frozen_random", "D_frozen_random_centroids_topk8_prerope"]:
                frozen_loss = loss

        if trained_loss is not None:
            print("\n" + "=" * 60)
            print("VAYUSPHERE DETAILED GAIN ANALYSIS")
            print("=" * 60)
            if baseline_loss is not None:
                gain_vs_baseline = baseline_loss - trained_loss
                print(f"Gain vs Baseline:                 {gain_vs_baseline:+.4f} (baseline: {baseline_loss:.4f} -> trained: {trained_loss:.4f})")
            if temp_loss is not None:
                gain_vs_temp = temp_loss - trained_loss
                print(f"Gain vs Learned Temperature:      {gain_vs_temp:+.4f} (temp: {temp_loss:.4f} -> trained: {trained_loss:.4f})")
            if frozen_loss is not None:
                gain_vs_frozen = frozen_loss - trained_loss
                print(f"Gain vs Frozen Random Centroids:  {gain_vs_frozen:+.4f} (frozen: {frozen_loss:.4f} -> trained: {trained_loss:.4f})")
                
                # trained_centroid_gain = loss_frozen_random - loss_trained
                trained_centroid_gain = frozen_loss - trained_loss
                print(f"Trained Centroid Gain:            {trained_centroid_gain:+.4f} (frozen: {frozen_loss:.4f} -> trained: {trained_loss:.4f})")

            if temp_loss is not None:
                # temp_control_gain = loss_learned_temp - loss_trained
                temp_control_gain = temp_loss - trained_loss
                print(f"Temp Control Gain:                {temp_control_gain:+.4f} (temp: {temp_loss:.4f} -> trained: {trained_loss:.4f})")

            # Find slowdown
            trained_row = df_results[df_results["name"] == trained_name]
            if not trained_row.empty:
                slowdown = trained_row.iloc[0].get(slowdown_col, 1.0)
                print(f"Throughput Slowdown vs Baseline:   {slowdown:.4f}x")
            print("=" * 60)

    # Confound Sweep Factorial Analysis
    if args.mode == "confound_sweep":
        def get_variant_loss(vname):
            row = df_results[df_results["name"] == vname]
            return row.iloc[0]["loss"] if not row.empty else None

        loss_all_post = get_variant_loss("scale_all_postrope")
        loss_topk_post = get_variant_loss("scale_topk8_postrope")
        loss_all_pre = get_variant_loss("scale_all_prerope")
        loss_topk_pre = get_variant_loss("scale_topk8_prerope")

        if all(x is not None for x in [loss_all_post, loss_topk_post, loss_all_pre, loss_topk_pre]):
            topk_post_effect = loss_topk_post - loss_all_post
            topk_pre_effect = loss_topk_pre - loss_all_pre
            prerope_all_effect = loss_all_pre - loss_all_post
            prerope_topk_effect = loss_topk_pre - loss_topk_post
            interaction = (loss_topk_pre - loss_all_pre) - (loss_topk_post - loss_all_post)

            factorial_df = pd.DataFrame([
                {
                    "effect_name": "topk_effect_under_post_rope",
                    "formula": "scale_topk8_postrope - scale_all_postrope",
                    "delta_loss": topk_post_effect
                },
                {
                    "effect_name": "topk_effect_under_pre_rope",
                    "formula": "scale_topk8_prerope - scale_all_prerope",
                    "delta_loss": topk_pre_effect
                },
                {
                    "effect_name": "prerope_effect_with_all_centroids",
                    "formula": "scale_all_prerope - scale_all_postrope",
                    "delta_loss": prerope_all_effect
                },
                {
                    "effect_name": "prerope_effect_with_topk8",
                    "formula": "scale_topk8_prerope - scale_topk8_postrope",
                    "delta_loss": prerope_topk_effect
                },
                {
                    "effect_name": "interaction_term",
                    "formula": "(topk_pre - all_pre) - (topk_post - all_post)",
                    "delta_loss": interaction
                }
            ])

            factorial_path = os.path.join(run_dir, "confound_factorial_analysis.csv")
            factorial_df.to_csv(factorial_path, index=False)
            print("\n" + "=" * 60)
            print("CONFOUND SWEEP FACTORIAL ANALYSIS")
            print("=" * 60)
            print(factorial_df.to_string(index=False))
            print("=" * 60)

    # Multi-seed Aggregation
    if args.mode in ["multi_seed", "proof_postrope_topk8"]:
        # Compute mean, std, mean delta, win count
        agg_rows = []

        # Calculate deltas for each seed run relative to baseline and controls
        for idx, row in df_results.iterrows():
            seed = row["seed"]
            # delta vs baseline
            baseline_row = df_results[(df_results["seed"] == seed) & (df_results["name"] == "A_baseline")]
            df_results.at[idx, "delta_vs_baseline"] = row["loss"] - baseline_row.iloc[0]["loss"] if not baseline_row.empty else 0.0
            # delta vs learned temp
            temp_row = df_results[(df_results["seed"] == seed) & (df_results["name"] == "B_learned_temp")]
            df_results.at[idx, "delta_vs_learned_temp"] = row["loss"] - temp_row.iloc[0]["loss"] if not temp_row.empty else 0.0
            # delta vs frozen random
            frozen_row = df_results[(df_results["seed"] == seed) & (df_results["name"] == "scale_topk8_postrope_frozen_random")]
            df_results.at[idx, "delta_vs_frozen_random"] = row["loss"] - frozen_row.iloc[0]["loss"] if not frozen_row.empty else 0.0
            # delta vs scale all postrope
            all_row = df_results[(df_results["seed"] == seed) & (df_results["name"] == "scale_all_postrope")]
            df_results.at[idx, "delta_vs_scale_all_postrope"] = row["loss"] - all_row.iloc[0]["loss"] if not all_row.empty else 0.0

        for name in [v[0] for v in variants]:
            variant_df = df_results[df_results["name"] == name]

            mean_loss = variant_df["loss"].mean()
            std_loss = variant_df["loss"].std()
            mean_ppl = variant_df["perplexity"].mean()
            mean_tps = variant_df["tokens_per_sec"].mean()
            mean_mem = variant_df["peak_mem_mb"].mean()
            mean_slowdown = variant_df["slowdown_vs_baseline"].mean()
            mean_delta = variant_df["delta_vs_baseline"].mean()

            # Calculate win counts vs baseline, temp, frozen
            wins_baseline = 0
            wins_temp = 0
            wins_frozen = 0
            for seed in [42, 123, 777]:
                v_row = variant_df[variant_df["seed"] == seed]
                
                b_row = df_results[(df_results["seed"] == seed) & (df_results["name"] == "A_baseline")]
                if not v_row.empty and not b_row.empty and v_row.iloc[0]["loss"] < b_row.iloc[0]["loss"]:
                    wins_baseline += 1
                    
                t_row = df_results[(df_results["seed"] == seed) & (df_results["name"] == "B_learned_temp")]
                if not v_row.empty and not t_row.empty and v_row.iloc[0]["loss"] < t_row.iloc[0]["loss"]:
                    wins_temp += 1
                    
                f_row = df_results[(df_results["seed"] == seed) & (df_results["name"] == "scale_topk8_postrope_frozen_random")]
                if not v_row.empty and not f_row.empty and v_row.iloc[0]["loss"] < f_row.iloc[0]["loss"]:
                    wins_frozen += 1

            agg_rows.append(
                {
                    "name": name,
                    "variant": name,
                    "mean_loss": mean_loss,
                    "std_loss": std_loss,
                    "mean_delta_vs_baseline": mean_delta,
                    "mean_delta_vs_learned_temp": variant_df["delta_vs_learned_temp"].mean() if "delta_vs_learned_temp" in variant_df.columns else 0.0,
                    "mean_delta_vs_frozen_random": variant_df["delta_vs_frozen_random"].mean() if "delta_vs_frozen_random" in variant_df.columns else 0.0,
                    "mean_delta_vs_scale_all_postrope": variant_df["delta_vs_scale_all_postrope"].mean() if "delta_vs_scale_all_postrope" in variant_df.columns else 0.0,
                    "win_count_vs_baseline": wins_baseline,
                    "win_count_vs_learned_temp": wins_temp,
                    "win_count_vs_frozen_random": wins_frozen,
                    "mean_tokens_per_sec": mean_tps,
                    "mean_slowdown_vs_baseline": mean_slowdown,
                    "mean_peak_mem_mb": mean_mem,
                    "mean_perplexity": mean_ppl,
                }
            )

        df_agg = pd.DataFrame(agg_rows)
        df_agg.to_csv(os.path.join(run_dir, "aggregate_mean_std.csv"), index=False)

        # Save proof_summary.csv
        proof_cols = [
            "variant",
            "mean_loss",
            "std_loss",
            "mean_delta_vs_baseline",
            "mean_delta_vs_learned_temp",
            "mean_delta_vs_frozen_random",
            "mean_delta_vs_scale_all_postrope",
            "win_count_vs_baseline",
            "win_count_vs_learned_temp",
            "win_count_vs_frozen_random",
            "mean_tokens_per_sec",
            "mean_slowdown_vs_baseline",
            "mean_peak_mem_mb"
        ]
        existing_cols = [c for c in proof_cols if c in df_agg.columns]
        df_proof = df_agg[existing_cols]
        df_proof.to_csv(os.path.join(run_dir, "proof_summary.csv"), index=False)

        print("\n" + "=" * 60)
        print("AGGREGATED MULTI-SEED PERFORMANCE REPORT")
        print("=" * 60)
        print(df_agg.to_string(index=False))
        print("=" * 60)

        # Detailed Gain Analysis (Aggregated Multi-Seed)
        baseline_mean = None
        temp_mean = None
        trained_mean = None
        frozen_mean = None
        all_mean = None
        trained_name = None
        
        for idx, row in df_agg.iterrows():
            name = row["name"]
            mean_loss = row["mean_loss"]
            if name == "A_baseline":
                baseline_mean = mean_loss
            elif name == "B_learned_temp":
                temp_mean = mean_loss
            elif name in ["scale_topk8_postrope", "D_vs_scale_topk8"]:
                trained_mean = mean_loss
                trained_name = name
            elif name in ["scale_topk8_postrope_frozen_random", "D_frozen_random_centroids_topk8_prerope"]:
                frozen_mean = mean_loss
            elif name == "scale_all_postrope":
                all_mean = mean_loss

        if trained_mean is not None:
            print("\n" + "=" * 60)
            print("VAYUSPHERE AGGREGATED DETAILED GAIN ANALYSIS")
            print("=" * 60)
            if baseline_mean is not None:
                gain_vs_baseline = baseline_mean - trained_mean
                print(f"Mean Gain vs Baseline:               {gain_vs_baseline:+.4f} (baseline: {baseline_mean:.4f} -> trained: {trained_mean:.4f})")
            if temp_mean is not None:
                gain_vs_temp = temp_mean - trained_mean
                print(f"Mean Gain vs Learned Temperature:    {gain_vs_temp:+.4f} (temp: {temp_mean:.4f} -> trained: {trained_mean:.4f})")
            if frozen_mean is not None:
                gain_vs_frozen = frozen_mean - trained_mean
                print(f"Mean Gain vs Frozen Random Centroids: {gain_vs_frozen:+.4f} (frozen: {frozen_mean:.4f} -> trained: {trained_mean:.4f})")
                
                # Trained centroid gain
                trained_centroid_gain = frozen_mean - trained_mean
                print(f"Mean Trained Centroid Gain:          {trained_centroid_gain:+.4f} (frozen: {frozen_mean:.4f} -> trained: {trained_mean:.4f})")

            if temp_mean is not None:
                temp_control_gain = temp_mean - trained_mean
                print(f"Mean Temp Control Gain:              {temp_control_gain:+.4f} (temp: {temp_mean:.4f} -> trained: {trained_mean:.4f})")

            # Find mean slowdown
            trained_row = df_agg[df_agg["name"] == trained_name]
            if not trained_row.empty:
                slowdown = trained_row.iloc[0].get("mean_slowdown_vs_baseline", 1.0)
                print(f"Mean Throughput Slowdown vs Baseline: {slowdown:.4f}x")
            print("=" * 60)

            # Claim Validation (Decision Rule)
            print("\n" + "=" * 60)
            print("VAYUSPHERE V0.2 CLAIM VALIDATION (DECISION RULE)")
            print("=" * 60)
            if all(x is not None for x in [trained_mean, baseline_mean, temp_mean, frozen_mean, all_mean]):
                beats_baseline = trained_mean < baseline_mean
                beats_temp = trained_mean < temp_mean
                beats_frozen = trained_mean < frozen_mean
                beats_all = trained_mean < all_mean
                
                print(f"1. Beats Baseline?                {beats_baseline} (trained: {trained_mean:.4f} vs baseline: {baseline_mean:.4f})")
                print(f"2. Beats Learned Temperature?      {beats_temp} (trained: {trained_mean:.4f} vs temp: {temp_mean:.4f})")
                print(f"3. Beats Frozen Random Centroids?  {beats_frozen} (trained: {trained_mean:.4f} vs frozen: {frozen_mean:.4f})")
                print(f"4. Beats Scale All Centroids?      {beats_all} (trained: {trained_mean:.4f} vs all: {all_mean:.4f})")
                
                if beats_baseline and beats_temp and beats_frozen and beats_all:
                    print("\n>>> CONCLUSION: CLAIM VALIDATED! VayuSphere scale_topk8_postrope outperforms all baselines and controls.")
                else:
                    print("\n>>> CONCLUSION: CLAIM NOT FULLY VALIDATED. One or more baselines/controls were not outperformed on average.")
            else:
                print("Could not evaluate decision rule: missing some variants in aggregation.")
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
                if common_steps is None:
                    common_steps = steps
                else:
                    common_steps = common_steps.intersection(steps)

            if common_steps:
                last_common_step = sorted(list(common_steps))[-1]
                baseline_val = metrics_dfs["A_baseline"][
                    metrics_dfs["A_baseline"]["step"] == last_common_step
                ]["eval_loss"].values[0]

                print(f"\nCOMMON STEP COMPARISON (Step {last_common_step})")
                print("-" * 60)
                print(f"{'Variant':<35} | {'Loss':<10} | {'Delta':<10} | {'Best':<10}")
                for name, _ in variants:
                    if name in metrics_dfs:
                        m_df = metrics_dfs[name]
                        current_val = m_df[m_df["step"] == last_common_step][
                            "eval_loss"
                        ].values[0]
                        best_val = m_df[m_df["step"] <= last_common_step][
                            "eval_loss"
                        ].min()
                        delta = current_val - baseline_val
                        print(
                            f"{name:<35} | {current_val:.4f}     | {delta:+.4f}    | {best_val:.4f}"
                        )
                print("-" * 60)


if __name__ == "__main__":
    main()
