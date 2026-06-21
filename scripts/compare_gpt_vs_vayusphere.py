import argparse
import logging
import torch
import os
import time
import datetime
import gc
import pandas as pd
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.train.config import TrainConfig
from astra_lm.train.trainer import Trainer
from astra_lm.train.kd_trainer import KDTrainer
from astra_lm.distill.teacher import load_teacher_model
from astra_lm.data.dataset import PretokenizedDataset
from astra_lm.data.collator import CausalLMCollator
from torch.utils.data import DataLoader
from astra_lm.utils import load_config_from_yaml
from astra_lm.utils.seed import set_seed


def clone_state_dict_to_cpu(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def make_initial_state_dict(model_config_path, seed):
    m_cfg = load_config_from_yaml(ModelConfig, model_config_path)
    set_seed(seed)
    model = DecoderForCausalLM(m_cfg)
    state = clone_state_dict_to_cpu(model)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


def run_experiment(
    name,
    model_config_path,
    train_config_path,
    data_dir,
    is_kd=False,
    teacher_config=None,
    alpha=0.5,
    temperature=2.0,
    teacher_dtype="8bit",
    output_dir=None,
    base_state_dict=None,
    seed=None,
    max_steps_override=None,
):
    logging.info(f"--- Starting Experiment: {name} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    m_cfg = load_config_from_yaml(ModelConfig, model_config_path)
    t_cfg = load_config_from_yaml(TrainConfig, train_config_path)

    # Set seed in train config
    if seed is not None:
        t_cfg.seed = seed

    if max_steps_override is not None:
        t_cfg.max_steps = max_steps_override

    if output_dir:
        t_cfg.output_dir = output_dir
    else:
        t_cfg.output_dir = os.path.join("experiments", name)

    t_cfg.overwrite_output_dir = True
    os.makedirs(t_cfg.output_dir, exist_ok=True)

    # Reset and log CUDA memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**2)
        logging.info(f"CUDA memory allocated before training starts: {mem_before:.2f} MB")

    # SEEDING: Set seed before model creation for reproducibility
    set_seed(t_cfg.seed)

    # Fair initialization: Create model and load state dict if provided
    model = DecoderForCausalLM(m_cfg)

    if base_state_dict:
        logging.info(f"Loading baseline state dict into {name} for fair comparison")
        missing, unexpected = model.load_state_dict(base_state_dict, strict=False)
        if unexpected:
            logging.warning(
                f"Unexpected keys when loading baseline into {name}: {unexpected}"
            )

        # Verify that only vayusphere centroids are missing
        vs_keys = [k for k in missing if "vayusphere" in k or "centroids" in k]
        other_keys = [k for k in missing if k not in vs_keys]
        if other_keys:
            logging.warning(f"Non-VayuSphere keys missing in {name}: {other_keys}")

    train_path = os.path.join(data_dir, "train.npy")
    val_path = os.path.join(data_dir, "val.npy")
    if not os.path.exists(val_path):
        val_path = os.path.join(data_dir, "validation.npy")

    train_dataset = PretokenizedDataset(train_path, seq_len=m_cfg.max_seq_len)
    eval_dataset = PretokenizedDataset(val_path, seq_len=m_cfg.max_seq_len)

    collator = CausalLMCollator()
    g = torch.Generator()
    g.manual_seed(t_cfg.seed)
    train_dl = DataLoader(
        train_dataset,
        batch_size=t_cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=g,
    )
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=t_cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    if is_kd:
        teacher = load_teacher_model(teacher_config, device=device, dtype=teacher_dtype)
        trainer = KDTrainer(
            student_model=model,
            teacher_model=teacher,
            train_config=t_cfg,
            train_dataloader=train_dl,
            eval_dataloader=eval_dl,
            alpha=alpha,
            temperature=temperature,
            device=device,
        )
    else:
        trainer = Trainer(
            model=model,
            train_config=t_cfg,
            train_dataloader=train_dl,
            eval_dataloader=eval_dl,
            device=device,
        )

    start_time = time.time()
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    # Get peak memory
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        logging.info(f"CUDA peak memory allocated during run (includes model): {peak_mem:.2f} MB")

    # Get final eval metrics from CSV
    metrics_path = os.path.join(t_cfg.output_dir, "metrics.csv")
    df = pd.read_csv(metrics_path)
    
    # Calculate number of eval points for warning verification
    eval_points = df["eval_loss"].dropna().count() if "eval_loss" in df.columns else 0
    
    if df.empty:
        final_loss = float("nan")
        final_ppl = 0.0
    else:
        final_loss = (
            df["eval_loss"].dropna().iloc[-1]
            if ("eval_loss" in df.columns and not df["eval_loss"].dropna().empty)
            else (df["loss"].dropna().iloc[-1] if ("loss" in df.columns and not df["loss"].dropna().empty) else float("nan"))
        )
        final_ppl = (
            df["eval_perplexity"].dropna().iloc[-1]
            if ("eval_perplexity" in df.columns and not df["eval_perplexity"].dropna().empty)
            else 0.0
        )

    params = sum(p.numel() for p in model.parameters())
    tokens_trained = trainer.total_tokens_trained
    
    # Clone state dict to CPU before destroying model
    cpu_state_dict = clone_state_dict_to_cpu(model)

    # Clean up model to free GPU memory immediately
    del model
    del trainer
    if is_kd:
        del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "loss": final_loss,
        "perplexity": final_ppl,
        "params": params,
        "peak_mem_mb": peak_mem,
        "total_time_sec": total_time,
        "tokens_per_sec": tokens_trained / total_time if total_time > 0 else 0,
        "max_steps": t_cfg.max_steps,
        "eval_points": eval_points,
        "state_dict": cpu_state_dict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--kd", action="store_true")
    parser.add_argument("--teacher", type=str, default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scratch_fair", "warm_start"],
        default="scratch_fair",
        help="Experiment mode: scratch_fair or warm_start",
    )
    parser.add_argument(
        "--warm_start_steps",
        type=int,
        default=10000,
        help="Number of steps for the initial warm-up baseline training in warm_start mode",
    )
    parser.add_argument(
        "--common_step_compare",
        action="store_true",
        help="Truncate comparison to the minimum common evaluation steps reached by all models",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_seed{args.seed}_{args.mode}"
    run_dir = os.path.join("outputs", "compare_gpt_vs_vayusphere", run_name)
    os.makedirs(run_dir, exist_ok=True)

    t_cfg = load_config_from_yaml(TrainConfig, args.train_config)
    max_steps = t_cfg.max_steps

    results = []

    if args.mode == "scratch_fair":
        logging.info(f"Running mode: scratch_fair (total steps: {max_steps})")
        # 1. Create the initial untrained state dict from baseline GPT config
        initial_state = make_initial_state_dict(
            "configs/model/gpt_nano_6gb.yaml",
            args.seed,
        )

        # 2. Train baseline GPT
        baseline_output_dir = os.path.join(run_dir, "baseline")
        res_gpt = run_experiment(
            "gpt_baseline",
            "configs/model/gpt_nano_6gb.yaml",
            args.train_config,
            args.data_dir,
            is_kd=args.kd,
            teacher_config=args.teacher,
            output_dir=baseline_output_dir,
            base_state_dict=initial_state,
            seed=args.seed,
        )
        # We don't need its state dict, so pop it to free CPU memory too
        res_gpt.pop("state_dict")
        results.append(res_gpt)

        # 3. Train VayuSphere GPT
        vs_output_dir = os.path.join(run_dir, "vayusphere")
        res_vs = run_experiment(
            "vayusphere_gpt",
            "configs/model/vayusphere_gpt_nano_6gb.yaml",
            args.train_config,
            args.data_dir,
            is_kd=args.kd,
            teacher_config=args.teacher,
            output_dir=vs_output_dir,
            base_state_dict=initial_state,
            seed=args.seed,
        )
        res_vs.pop("state_dict")
        results.append(res_vs)

        # Clean up initial state dict
        del initial_state
        gc.collect()

    elif args.mode == "warm_start":
        logging.info(f"Running mode: warm_start")
        warmup_steps = args.warm_start_steps
        if warmup_steps >= max_steps:
            raise ValueError(
                f"--warm_start_steps ({warmup_steps}) must be less than training max_steps ({max_steps})"
            )

        continued_steps = max_steps - warmup_steps
        logging.info(
            f"Phase 1: Warmup GPT Baseline for {warmup_steps} steps"
        )

        warmup_output_dir = os.path.join(run_dir, "gpt_warmup")
        res_warmup = run_experiment(
            "gpt_warmup",
            "configs/model/gpt_nano_6gb.yaml",
            args.train_config,
            args.data_dir,
            is_kd=args.kd,
            teacher_config=args.teacher,
            output_dir=warmup_output_dir,
            seed=args.seed,
            max_steps_override=warmup_steps,
        )
        warmup_state_dict = res_warmup.pop("state_dict")

        logging.info(
            f"Phase 2: Continued training for {continued_steps} steps starting from warmup checkpoint"
        )

        # A) Continue baseline GPT
        baseline_output_dir = os.path.join(run_dir, "baseline_continued")
        res_gpt = run_experiment(
            "gpt_continued",
            "configs/model/gpt_nano_6gb.yaml",
            args.train_config,
            args.data_dir,
            is_kd=args.kd,
            teacher_config=args.teacher,
            output_dir=baseline_output_dir,
            base_state_dict=warmup_state_dict,
            seed=args.seed,
            max_steps_override=continued_steps,
        )
        res_gpt.pop("state_dict")
        results.append(res_gpt)

        # B) Convert and train VayuSphere GPT
        vs_output_dir = os.path.join(run_dir, "vayusphere_continued")
        res_vs = run_experiment(
            "vayusphere_continued",
            "configs/model/vayusphere_gpt_nano_6gb.yaml",
            args.train_config,
            args.data_dir,
            is_kd=args.kd,
            teacher_config=args.teacher,
            output_dir=vs_output_dir,
            base_state_dict=warmup_state_dict,
            seed=args.seed,
            max_steps_override=continued_steps,
        )
        res_vs.pop("state_dict")
        results.append(res_vs)

        # Clean up warmup state dict from CPU
        del warmup_state_dict
        gc.collect()

    # Report
    print("\n" + "=" * 50)
    print("COMPARISON REPORT")
    print("=" * 50)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Save CSV inside run_dir
    csv_path = os.path.join(run_dir, "comparison_results.csv")
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Saved comparison report to {csv_path}")

    # Check for warnings and compute common step metrics
    if len(results) >= 2:
        warnings = []

        # 1. Check for step mismatches
        max_steps_list = [r["max_steps"] for r in results]
        eval_points_list = [r["eval_points"] for r in results]

        if len(set(max_steps_list)) > 1:
            warnings.append(f"WARNING: Models have different max_steps: {max_steps_list}")
        if len(set(eval_points_list)) > 1:
            warnings.append(f"WARNING: Models have different evaluation points: {eval_points_list}")

        # 2. Common step comparison logic
        if args.common_step_compare or True: # Always show if we have baseline
            # Load all metrics.csv
            metrics_dfs = {}
            for r in results:
                m_path = os.path.join(run_dir, r["name"].replace("_gpt", "").replace("gpt_", ""), "metrics.csv")
                # Handle path differences
                actual_m_path = None
                possible_paths = [
                    os.path.join(run_dir, r["name"], "metrics.csv"),
                    os.path.join(run_dir, "baseline", "metrics.csv") if "baseline" in r["name"] else None,
                    os.path.join(run_dir, "vayusphere", "metrics.csv") if "vayusphere" in r["name"] else None,
                    os.path.join(run_dir, "gpt_continued", "metrics.csv") if "continued" in r["name"] and "gpt" in r["name"] else None,
                    os.path.join(run_dir, "vayusphere_continued", "metrics.csv") if "continued" in r["name"] and "vayusphere" in r["name"] else None,
                ]
                for p in possible_paths:
                    if p and os.path.exists(p):
                        actual_m_path = p
                        break

                if actual_m_path:
                    m_df = pd.read_csv(actual_m_path)
                    m_df = m_df[m_df["eval_loss"].notna()]
                    metrics_dfs[r["name"]] = m_df

            if len(metrics_dfs) >= 2:
                # Find common steps
                common_steps = None
                for name, m_df in metrics_dfs.items():
                    steps = set(m_df["step"].tolist())
                    if common_steps is None:
                        common_steps = steps
                    else:
                        common_steps = common_steps.intersection(steps)

                if common_steps:
                    common_steps = sorted(list(common_steps))
                    last_common_step = common_steps[-1]
                    logging.info(f"Comparing at common step: {last_common_step}")

                    print("\n" + "-" * 50)
                    print(f"COMMON STEP COMPARISON (Step {last_common_step})")
                    print("-" * 50)

                    baseline_name = results[0]["name"]
                    baseline_val = metrics_dfs[baseline_name][metrics_dfs[baseline_name]["step"] == last_common_step]["eval_loss"].values[0]

                    print(f"{'Model':<25} | {'Eval Loss':<10} | {'Delta':<10} | {'Best (Common)':<10}")
                    for r in results:
                        name = r["name"]
                        if name in metrics_dfs:
                            m_df = metrics_dfs[name]
                            common_df = m_df[m_df["step"] <= last_common_step]
                            current_val = m_df[m_df["step"] == last_common_step]["eval_loss"].values[0]
                            best_val = common_df["eval_loss"].min()
                            delta = current_val - baseline_val
                            print(f"{name:<25} | {current_val:.4f}     | {delta:+.4f}    | {best_val:.4f}")
                    print("-" * 50)
                else:
                    warnings.append("WARNING: No common evaluation steps found for comparison.")

        if warnings:
            print("\n" + "!" * 50)
            print("EXPERIMENT WARNINGS")
            print("!" * 50)
            for w in warnings:
                print(w)
            print("!" * 50 + "\n")


if __name__ == "__main__":
    main()
