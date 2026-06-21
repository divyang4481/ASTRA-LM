import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import logging
import time

from .config import TrainConfig
from .optimizer import create_optimizer, get_cosine_schedule_with_warmup
from .checkpoint import save_checkpoint
from ..eval.perplexity import evaluate_perplexity

logger = logging.getLogger(__name__)

class Trainer:
    """
    A simple, config-driven trainer for the ASTRA-LM model.
    """
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.config = train_config
        self.device = torch.device(device)

        # Environment diagnostics
        logger.info(f"Torch Version: {torch.__version__}")
        if hasattr(torch.version, "cuda"):
            logger.info(f"CUDA Build: {torch.version.cuda}")

        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")

        if self.device.type == "cuda":
            if not cuda_available:
                raise RuntimeError("CUDA device requested but CUDA is not available.")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Precision setup
        self.autocast_dtype = torch.float32
        self.use_amp = False
        self.scaler = None

        if self.config.mixed_precision != "none":
            if self.device.type == "cuda":
                if self.config.mixed_precision == "bf16":
                    if torch.cuda.is_bf16_supported():
                        self.autocast_dtype = torch.bfloat16
                        self.use_amp = True
                        logger.info("Using bf16 mixed precision")
                    else:
                        logger.warning("bf16 requested but not supported by GPU. Falling back to fp16.")
                        self.autocast_dtype = torch.float16
                        self.use_amp = True
                        self.scaler = torch.cuda.amp.GradScaler()
                        logger.info("Using fp16 mixed precision with GradScaler")
                elif self.config.mixed_precision == "fp16":
                    self.autocast_dtype = torch.float16
                    self.use_amp = True
                    self.scaler = torch.cuda.amp.GradScaler()
                    logger.info("Using fp16 mixed precision with GradScaler")
                elif self.config.mixed_precision == "auto":
                    if torch.cuda.is_bf16_supported():
                        self.autocast_dtype = torch.bfloat16
                        self.use_amp = True
                        logger.info("Using auto-selected bf16 mixed precision")
                    else:
                        self.autocast_dtype = torch.float16
                        self.use_amp = True
                        self.scaler = torch.cuda.amp.GradScaler()
                        logger.info("Using auto-selected fp16 mixed precision with GradScaler")
            else:
                logger.warning(f"Mixed precision {self.config.mixed_precision} requested on CPU, which is not supported in this trainer. Using float32.")

        # Set up optimizer and scheduler
        self.optimizer = create_optimizer(
            model=self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
            min_lr_ratio=self.config.min_lr_ratio
        )

        self.global_step = 0

    def train(self):
        """
        Runs the training loop.
        """
        logger.info("Starting training...")
        logger.info(f"Total steps: {self.config.max_steps}")
        logger.info(f"Device: {self.device}")

        self.model.train()

        # Create an infinite iterator for the training dataloader
        def get_train_batch():
            while True:
                for batch in self.train_dataloader:
                    yield batch

        train_iter = iter(get_train_batch())

        start_time = time.time()
        tokens_processed = 0
        
        start_step = self.global_step + 1
        
        for step in range(start_step, self.config.max_steps + 1):
            self.global_step = step
            batch = next(train_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device) if "labels" in batch else input_ids

            # Forward pass with AMP
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (only at accumulation boundary)
            if step % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            # Track tokens
            batch_size, seq_len = input_ids.shape
            tokens_processed += batch_size * seq_len

            # Logging
            if step % self.config.logging_steps == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                current_lr = self.scheduler.get_last_lr()[0]

                # Aggregate candidate ratios across all layers
                candidate_ratios = []
                if "diagnostics" in outputs:
                    candidate_ratios = [
                        diag["candidate_ratio"] 
                        for diag in outputs["diagnostics"] 
                        if diag and "candidate_ratio" in diag
                    ]
                
                diag_str = ""
                if candidate_ratios:
                    avg_ratio = sum(candidate_ratios) / len(candidate_ratios)
                    diag_str = f" | Candidate Ratio: {avg_ratio:.2%}"

                mem_str = ""
                if self.device.type == "cuda":
                    allocated = torch.cuda.memory_allocated() / (1024**2)
                    reserved = torch.cuda.memory_reserved() / (1024**2)
                    mem_str = f" | Mem: {allocated:.0f}/{reserved:.0f}MB"

                logger.info(
                    f"Step {step}/{self.config.max_steps} | "
                    f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}{diag_str}{mem_str} | "
                    f"LR: {current_lr:.2e} | "
                    f"Tok/s: {tokens_per_sec:.0f}"
                )

                # Reset tracking for next log interval
                start_time = time.time()
                tokens_processed = 0

            # Evaluation
            if step % self.config.eval_steps == 0 and self.eval_dataloader is not None:
                self._evaluate()

            # Checkpointing
            if step % self.config.save_steps == 0:
                save_checkpoint(
                    output_dir=self.config.output_dir,
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=self.config
                )

        logger.info("Training complete!")

    def _evaluate(self):
        """
        Runs the evaluation loop.
        """
        logger.info(f"Running evaluation at step {self.global_step}...")

        eval_results = evaluate_perplexity(
            model=self.model,
            dataloader=self.eval_dataloader,
            device=self.device,
            max_batches=self.config.max_eval_batches
        )

        logger.info(
            f"Eval Step {self.global_step} | "
            f"Loss: {eval_results['loss']:.4f} | "
            f"Perplexity: {eval_results['perplexity']:.4f}"
        )

        self.model.train()
