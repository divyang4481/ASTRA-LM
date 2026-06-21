import argparse
import logging
import torch
import os
import time
import datetime
import gc
import copy
import pandas as pd
import yaml
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.train.config import TrainConfig
from astra_lm.train.trainer import Trainer
from astra_lm.data.dataset import PretokenizedDataset
from astra_lm.data.collator import CausalLMCollator
from torch.utils.data import DataLoader
from astra_lm.utils import load_config_from_yaml
from astra_lm.utils.seed import set_seed

def run_experiment(
    name,
    m_cfg,
    t_cfg,
    data_dir,
    output_dir,
    base_state_dict=None
):
    logging.info(f"--- Starting Ablation: {name} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t_cfg.output_dir = output_dir
    os.makedirs(t_cfg.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # SEEDING: Set seed before model creation for reproducibility
    set_seed(t_cfg.seed)

    model = DecoderForCausalLM(m_cfg)

    if base_state_dict:
        logging.info(f"Loading baseline state dict for fair ablation")
        model.load_state_dict(base_state_dict, strict=False)

    train_path = os.path.join(data_dir, "train.npy")
    val_path = os.path.join(data_dir, "val.npy")
    if not os.path.exists(val_path):
        val_path = os.path.join(data_dir, "validation.npy")

    train_dataset = PretokenizedDataset(train_path, seq_len=m_cfg.max_seq_len)
    eval_dataset = PretokenizedDataset(val_path, seq_len=m_cfg.max_seq_len)

    collator = CausalLMCollator()
    g = torch.Generator()
    g.manual_seed(t_cfg.seed)
    train_dl = DataLoader(train_dataset, batch_size=t_cfg.per_device_train_batch_size, shuffle=True, collate_fn=collator, generator=g)
    eval_dl = DataLoader(eval_dataset, batch_size=t_cfg.per_device_eval_batch_size, shuffle=False, collate_fn=collator)

    trainer = Trainer(
        model=model,
        train_config=t_cfg,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        device=device
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    # Get peak memory
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

    # Get final eval metrics from CSV
    metrics_path = os.path.join(t_cfg.output_dir, "metrics.csv")
    df = pd.read_csv(metrics_path)

    final_eval_loss = df['eval_loss'].dropna().iloc[-1] if not df['eval_loss'].dropna().empty else None
    final_train_loss = df['loss'].dropna().iloc[-1] if not df['loss'].dropna().empty else None
    final_ppl = df['eval_perplexity'].dropna().iloc[-1] if not df['eval_perplexity'].dropna().empty else 0

    # Gate stats
    avg_q_gate = df['vs_q_gate_mean'].dropna().mean() if 'vs_q_gate_mean' in df else None
    avg_k_gate = df['vs_k_gate_mean'].dropna().mean() if 'vs_k_gate_mean' in df else None
    avg_grad_norm = df['vs_centroid_grad_norm_mean'].dropna().mean() if 'vs_centroid_grad_norm_mean' in df else None

    params = sum(p.numel() for p in model.parameters())

    res = {
        "name": name,
        "target": m_cfg.vayusphere_target,
        "alpha": m_cfg.vayusphere_alpha,
        "centroids": m_cfg.vayusphere_num_centroids,
        "eval_loss": final_eval_loss,
        "train_loss": final_train_loss,
        "perplexity": final_ppl,
        "params": params,
        "peak_mem_mb": peak_mem,
        "tokens_per_sec": (t_cfg.max_steps * t_cfg.per_device_train_batch_size * m_cfg.max_seq_len) / total_time,
        "avg_q_gate": avg_q_gate,
        "avg_k_gate": avg_k_gate,
        "avg_centroid_grad": avg_grad_norm
    }

    # Cleanup
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res

def main():
    parser = argparse.ArgumentParser(description="VayuSphere Ablation Runner")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--base_model_config", type=str, default="configs/model/gpt_nano_6gb.yaml")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--max_eval_batches", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", "ablate_vayusphere", f"{timestamp}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Load base configs
    m_cfg_base = load_config_from_yaml(ModelConfig, args.base_model_config)
    t_cfg_base = load_config_from_yaml(TrainConfig, args.train_config)
    t_cfg_base.seed = args.seed
    if args.max_steps:
        t_cfg_base.max_steps = args.max_steps
    if args.max_eval_batches:
        t_cfg_base.max_eval_batches = args.max_eval_batches

    # 1. Baseline Run to get initial weights
    baseline_m_cfg = yaml.full_load(yaml.dump(m_cfg_base))
    baseline_m_cfg = ModelConfig(**baseline_m_cfg)
    baseline_m_cfg.use_vayusphere = False

    # Capture the exact state dict after seeded initialization for all experiments
    set_seed(args.seed)
    temp_model = DecoderForCausalLM(baseline_m_cfg)
    base_state_dict = copy.deepcopy(temp_model.state_dict())
    del temp_model
    gc.collect()

    # Run baseline experiment using the captured state dict
    res_baseline = run_experiment(
        "baseline",
        baseline_m_cfg,
        t_cfg_base,
        args.data_dir,
        os.path.join(run_dir, "baseline"),
        base_state_dict=base_state_dict
    )

    results = [res_baseline]

    # Ablation Grid
    targets = ["q", "k", "qk"]
    alphas = [0.01, 0.03, 0.05, 0.1]
    num_centroids = [8, 16, 32]

    for target in targets:
        for alpha in alphas:
            for centroids in num_centroids:
                name = f"{target}_a{alpha}_c{centroids}"

                m_cfg = yaml.full_load(yaml.dump(m_cfg_base))
                m_cfg = ModelConfig(**m_cfg)
                m_cfg.use_vayusphere = True
                m_cfg.vayusphere_target = target
                m_cfg.vayusphere_alpha = alpha
                m_cfg.vayusphere_num_centroids = centroids

                res = run_experiment(
                    name,
                    m_cfg,
                    t_cfg_base,
                    args.data_dir,
                    os.path.join(run_dir, name),
                    base_state_dict=base_state_dict
                )
                results.append(res)

                # Save intermediate results
                pd.DataFrame(results).to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    print("\n" + "="*50)
    print("ABLATION REPORT")
    print("="*50)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
