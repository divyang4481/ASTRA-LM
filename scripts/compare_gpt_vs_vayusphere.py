import argparse
import logging
import torch
import os
import time
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

def run_experiment(
    name,
    model_config_path,
    train_config_path,
    data_dir,
    is_kd=False,
    teacher_config=None,
    alpha=0.5,
    temperature=2.0,
    teacher_dtype="8bit"
):
    logging.info(f"--- Starting Experiment: {name} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    m_cfg = load_config_from_yaml(ModelConfig, model_config_path)
    t_cfg = load_config_from_yaml(TrainConfig, train_config_path)

    t_cfg.output_dir = os.path.join("experiments", name)
    os.makedirs(t_cfg.output_dir, exist_ok=True)

    model = DecoderForCausalLM(m_cfg)

    train_path = os.path.join(data_dir, "train.npy")
    val_path = os.path.join(data_dir, "val.npy")
    if not os.path.exists(val_path):
        val_path = os.path.join(data_dir, "validation.npy")

    train_dataset = PretokenizedDataset(train_path, seq_len=m_cfg.max_seq_len)
    eval_dataset = PretokenizedDataset(val_path, seq_len=m_cfg.max_seq_len)

    collator = CausalLMCollator()
    train_dl = DataLoader(train_dataset, batch_size=t_cfg.per_device_train_batch_size, shuffle=True, collate_fn=collator)
    eval_dl = DataLoader(eval_dataset, batch_size=t_cfg.per_device_eval_batch_size, shuffle=False, collate_fn=collator)

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
            device=device
        )
    else:
        trainer = Trainer(
            model=model,
            train_config=t_cfg,
            train_dataloader=train_dl,
            eval_dataloader=eval_dl,
            device=device
        )

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    # Get peak memory
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

    # Get final eval metrics from CSV
    metrics_path = os.path.join(t_cfg.output_dir, "metrics.csv")
    df = pd.read_csv(metrics_path)
    final_loss = df['eval_loss'].dropna().iloc[-1] if not df['eval_loss'].dropna().empty else df['loss'].iloc[-1]
    final_ppl = df['eval_perplexity'].dropna().iloc[-1] if not df['eval_perplexity'].dropna().empty else 0

    params = sum(p.numel() for p in model.parameters())

    return {
        "name": name,
        "loss": final_loss,
        "perplexity": final_ppl,
        "params": params,
        "peak_mem_mb": peak_mem,
        "total_time_sec": total_time,
        "tokens_per_sec": (t_cfg.max_steps * t_cfg.per_device_train_batch_size * m_cfg.max_seq_len) / total_time
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--kd", action="store_true")
    parser.add_argument("--teacher", type=str, default="gpt2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = []

    # 1. Baseline GPT
    res_gpt = run_experiment(
        "gpt_baseline",
        "configs/model/gpt_nano_6gb.yaml",
        args.train_config,
        args.data_dir,
        is_kd=args.kd,
        teacher_config=args.teacher
    )
    results.append(res_gpt)

    # 2. VayuSphere GPT
    res_vs = run_experiment(
        "vayusphere_gpt",
        "configs/model/vayusphere_gpt_nano_6gb.yaml",
        args.train_config,
        args.data_dir,
        is_kd=args.kd,
        teacher_config=args.teacher
    )
    results.append(res_vs)

    # Report
    print("\n" + "="*50)
    print("COMPARISON REPORT")
    print("="*50)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    df_results.to_csv("comparison_results.csv", index=False)

if __name__ == "__main__":
    main()
