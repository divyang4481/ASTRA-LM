import argparse
import logging
import torch
from torch.utils.data import DataLoader

from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.train.config import TrainConfig
import os
from astra_lm.train.kd_trainer import KDTrainer
from astra_lm.distill.teacher import load_teacher_model
from astra_lm.data.dataset import SyntheticDataset, PretokenizedDataset
from astra_lm.data.collator import CausalLMCollator
from astra_lm.utils import load_config_from_yaml

def main():
    parser = argparse.ArgumentParser(description="KD Training for ASTRA-LM")
    parser.add_argument("--student_config", type=str, required=True, help="Path to student model config YAML")
    parser.add_argument("--teacher_config", type=str, required=True, help="Path to teacher model config YAML")
    parser.add_argument("--teacher_checkpoint", type=str, help="Path to teacher model checkpoint (.pt)")
    parser.add_argument("--train_config", type=str, required=True, help="Path to train config YAML")
    parser.add_argument("--alpha", type=float, default=0.5, help="KD loss weight")
    parser.add_argument("--temperature", type=float, default=2.0, help="KD temperature")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--data_dir", type=str, help="Path to directory with train.npy and val.npy")
    parser.add_argument("--allow_random_teacher", action="store_true", help="Allow random teacher if checkpoint missing")
    parser.add_argument("--teacher_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16", "8bit", "4bit"], help="Data type or quantization for loading HF teacher model")
    parser.add_argument("--batch_size", type=int, help="Override per_device_train_batch_size")
    parser.add_argument("--seq_len", type=int, help="Override max_seq_len for KD")
    parser.add_argument("--grad_accum", type=int, help="Override gradient_accumulation_steps")
    
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configs
    logger.info(f"Loading student config from {args.student_config}")
    s_m_cfg = load_config_from_yaml(ModelConfig, args.student_config)
    
    logger.info(f"Loading train config from {args.train_config}")
    t_cfg = load_config_from_yaml(TrainConfig, args.train_config)
    
    if args.output_dir:
        t_cfg.output_dir = args.output_dir

    if args.batch_size:
        t_cfg.per_device_train_batch_size = args.batch_size
        t_cfg.per_device_eval_batch_size = args.batch_size

    if args.grad_accum:
        t_cfg.gradient_accumulation_steps = args.grad_accum

    # Initialize student model
    logger.info("Initializing student model...")
    student_model = DecoderForCausalLM(s_m_cfg)

    # Initialize teacher model
    is_local_teacher_config = (
        os.path.exists(args.teacher_config)
        or args.teacher_config.endswith(".yaml")
        or args.teacher_config.endswith(".yml")
        or args.teacher_config.endswith(".json")
    )

    if is_local_teacher_config:
        logger.info(f"Teacher source: local_config ({args.teacher_config})")
        if not args.teacher_checkpoint and not args.allow_random_teacher:
            raise ValueError(f"Local teacher config {args.teacher_config} requires --teacher_checkpoint. Use --allow_random_teacher for debugging only.")
    else:
        logger.info(f"Teacher source: huggingface ({args.teacher_config})")

    logger.info(f"Loading teacher model from {args.teacher_config} in dtype/quant: {args.teacher_dtype}...")
    teacher_model = load_teacher_model(
        config_path=args.teacher_config,
        checkpoint_path=args.teacher_checkpoint,
        device=device,
        dtype=args.teacher_dtype
    )

    # Determine max sequence length for KD
    teacher_seq_len = 1024
    if hasattr(teacher_model, "config"):
        if hasattr(teacher_model.config, "n_positions"):
            teacher_seq_len = teacher_model.config.n_positions
        elif hasattr(teacher_model.config, "max_position_embeddings"):
            teacher_seq_len = teacher_model.config.max_position_embeddings
        elif hasattr(teacher_model.config, "max_seq_len"):
            teacher_seq_len = teacher_model.config.max_seq_len

    max_kd_seq_len = min(s_m_cfg.max_seq_len, teacher_seq_len)
    if args.seq_len:
        max_kd_seq_len = min(max_kd_seq_len, args.seq_len)

    logger.info(f"Setting KD sequence length to {max_kd_seq_len} (Student: {s_m_cfg.max_seq_len}, Teacher: {teacher_seq_len})")

    # Initialize dataset
    if args.data_dir:
        logger.info(f"Initializing dataset from dir: {args.data_dir}")
        train_path = os.path.join(args.data_dir, "train.npy")
        val_path = os.path.join(args.data_dir, "val.npy")
        if not os.path.exists(val_path):
            val_path = os.path.join(args.data_dir, "validation.npy")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}.")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found. Expected val.npy or validation.npy.")

        train_dataset = PretokenizedDataset(train_path, seq_len=max_kd_seq_len)
        eval_dataset = PretokenizedDataset(val_path, seq_len=max_kd_seq_len)
    else:
        logger.info("Initializing synthetic dataset...")
        train_dataset = SyntheticDataset(
            vocab_size=s_m_cfg.vocab_size,
            seq_len=max_kd_seq_len,
            num_samples=1000
        )
        eval_dataset = SyntheticDataset(
            vocab_size=s_m_cfg.vocab_size,
            seq_len=max_kd_seq_len,
            num_samples=100
        )

    collator = CausalLMCollator()
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=t_cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=t_cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator
    )

    # Initialize KD Trainer
    trainer = KDTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        train_config=t_cfg,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        alpha=args.alpha,
        temperature=args.temperature,
        device=device
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
