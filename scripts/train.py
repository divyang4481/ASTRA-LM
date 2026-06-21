import argparse
import logging
import os
from torch.utils.data import DataLoader

from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.train.config import TrainConfig, DataConfig
from astra_lm.train.trainer import Trainer
from astra_lm.data.dataset import SyntheticDataset, PretokenizedDataset
from astra_lm.data.collator import CausalLMCollator
from astra_lm.utils import load_config_from_yaml

def main():
    parser = argparse.ArgumentParser(description="Train ASTRA-LM")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--train_config", type=str, required=True, help="Path to train config YAML")
    parser.add_argument("--data_config", type=str, help="Path to data config YAML (optional)")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--data_dir", type=str, help="Path to directory with train.npy and val.npy")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda or cpu)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum training steps")
    
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    import torch
    cuda_available = torch.cuda.is_available()
    logger.info(f"Torch Version: {torch.__version__}")
    logger.info(f"CUDA Available at script start: {cuda_available}")
    if cuda_available:
        logger.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    device = args.device if args.device else ("cuda" if cuda_available else "cpu")
    logger.info(f"Using device: {device}")

    # Load configs
    logger.info(f"Loading model config from {args.model_config}")
    m_cfg = load_config_from_yaml(ModelConfig, args.model_config)
    
    logger.info(f"Loading train config from {args.train_config}")
    t_cfg = load_config_from_yaml(TrainConfig, args.train_config)
    
    if args.data_config:
        logger.info(f"Loading data config from {args.data_config}")
        d_cfg = load_config_from_yaml(DataConfig, args.data_config)
    else:
        logger.info("Using default data config (synthetic)")
        d_cfg = DataConfig(vocab_size=m_cfg.vocab_size, seq_len=m_cfg.max_seq_len)

    # Overrides
    if args.max_steps is not None:
        t_cfg.max_steps = args.max_steps
        logger.info(f"Overriding max_steps to {t_cfg.max_steps}")

    if args.output_dir:
        t_cfg.output_dir = args.output_dir

    # Initialize model
    logger.info("Initializing model...")
    model = DecoderForCausalLM(m_cfg)

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

        train_dataset = PretokenizedDataset(train_path, seq_len=m_cfg.max_seq_len)
        eval_dataset = PretokenizedDataset(val_path, seq_len=m_cfg.max_seq_len)
    else:
        logger.info("Initializing synthetic dataset...")
        train_dataset = SyntheticDataset(
            vocab_size=m_cfg.vocab_size,
            seq_len=m_cfg.max_seq_len,
            num_samples=1000
        )
        eval_dataset = SyntheticDataset(
            vocab_size=m_cfg.vocab_size,
            seq_len=m_cfg.max_seq_len,
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

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_config=t_cfg,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device
    )

    # Handle Resumption
    if args.resume_from:
        from astra_lm.train.checkpoint import load_checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_data = load_checkpoint(args.resume_from, model, optimizer=trainer.optimizer, scheduler=trainer.scheduler, map_location=device)
        trainer.global_step = checkpoint_data["step"]

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
