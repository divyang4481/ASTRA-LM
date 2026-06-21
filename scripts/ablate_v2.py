import argparse
import logging
import os
import datetime
import yaml
import pandas as pd
from scripts.compare_gpt_vs_vayusphere import run_experiment, make_initial_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{timestamp}_seed{args.seed}"

    run_dir = os.path.join("outputs", "ablate_v2", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Base config and state dict
    base_model_config = "configs/model/gpt_nano_6gb.yaml"
    initial_state = make_initial_state_dict(base_model_config, args.seed)

    def get_config(updates):
        with open(base_model_config, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg.update(updates)
        tmp_path = os.path.join(run_dir, "tmp_config.yaml")
        with open(tmp_path, 'w') as f:
            yaml.dump(cfg, f)
        return tmp_path

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
        ("F_vs_tangent_scale_pre_rope_topk8", {
            "use_vayusphere": True,
            "vayusphere_mode": "tangent_scale",
            "vayusphere_alpha": 0.1,
            "vayusphere_scale_alpha": 0.1,
            "vayusphere_topk_centroids": 8,
            "vayusphere_apply_stage": "pre_rope"
        }),
    ]

    results = []
    for name, updates in variants:
        cfg_path = get_config(updates)
        res = run_experiment(
            name=name,
            model_config_path=cfg_path,
            train_config_path=args.train_config,
            data_dir=args.data_dir,
            output_dir=os.path.join(run_dir, name),
            base_state_dict=initial_state,
            seed=args.seed,
            max_steps_override=args.max_steps
        )
        res.pop("state_dict")
        results.append(res)

    # Final report using the logic from compare script
    print("\n" + "=" * 60)
    print("VAYUSPHERE V0.2 ABLATION REPORT")
    print("=" * 60)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    df_results.to_csv(os.path.join(run_dir, "ablation_results.csv"), index=False)

    # Common step comparison
    metrics_dfs = {}
    for r in results:
        m_path = os.path.join(run_dir, r["name"], "metrics.csv")
        if os.path.exists(m_path):
            m_df = pd.read_csv(m_path)
            metrics_dfs[r["name"]] = m_df[m_df["eval_loss"].notna()]

    if len(metrics_dfs) >= 2:
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
