import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import logging
import time

from .trainer import Trainer
from .config import TrainConfig
from ..distill.kd_losses import kl_distillation_loss, kl_topk_distillation_loss

logger = logging.getLogger(__name__)

class KDTrainer(Trainer):
    """
    Knowledge Distillation Trainer for student DHRUVA models.
    """
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_config: TrainConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        alpha: float = 0.5,        # Weight for KD loss
        temperature: float = 2.0,  # KD temperature
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(
            model=student_model,
            train_config=train_config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            device=device
        )
        self.teacher_model = teacher_model.to(self.device).eval()
        self.alpha = alpha
        self.temperature = temperature

    def train(self):
        """
        Modified training loop with KD and CHAKRA routing logs.
        """
        logger.info("Starting KD training...")
        logger.info(f"Alpha: {self.alpha}, Temperature: {self.temperature}")
        
        self.model.train()
        
        def get_train_batch():
            while True:
                for batch in self.train_dataloader:
                    yield batch

        train_iter = iter(get_train_batch())
        start_time = time.time()
        tokens_processed = 0

        for step in range(1, self.config.max_steps + 1):
            self.global_step = step
            batch = next(train_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device) if "labels" in batch else input_ids

            # Forward pass with AMP
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp):
                # Student forward
                student_outputs = self.model(input_ids=input_ids, labels=labels)
                student_logits = student_outputs["logits"]
                ce_loss = student_outputs["loss"]

                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(input_ids=input_ids)
                    teacher_logits = teacher_outputs["logits"]

                # Perform Top-K distillation to prevent CUDA OOM on full-vocabulary logits tensors
                # student/teacher logits: [batch_size, seq_len, vocab_size] (e.g. 4 * 1024 * 50257 = 205,827,072 floats)
                # Allocating full probability/log-probability tensors for F.kl_div uses massive memory.
                # Restricting to top 100 logit values is mathematicaly equivalent for KD while reducing VRAM memory allocation by 500x.
                k = min(100, teacher_logits.size(-1))
                teacher_topk_values, teacher_topk_indices = torch.topk(teacher_logits, k=k, dim=-1)
                
                kd_loss = kl_topk_distillation_loss(
                    student_logits=student_logits,
                    teacher_topk_indices=teacher_topk_indices,
                    teacher_topk_values=teacher_topk_values,
                    temperature=self.temperature
                )

                # Combined loss
                loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
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
                if "diagnostics" in student_outputs:
                    candidate_ratios = [
                        diag["candidate_ratio"] 
                        for diag in student_outputs["diagnostics"] 
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
                    f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} (CE: {ce_loss.item():.4f}, KD: {kd_loss.item():.4f}){diag_str}{mem_str} | "
                    f"LR: {current_lr:.2e} | Tok/s: {tokens_per_sec:.0f}"
                )
                start_time = time.time()
                tokens_processed = 0

            # Evaluation
            if step % self.config.eval_steps == 0 and self.eval_dataloader is not None:
                self._evaluate()

            # Checkpointing
            if step % self.config.save_steps == 0:
                from .checkpoint import save_checkpoint
                save_checkpoint(
                    output_dir=self.config.output_dir,
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=self.config
                )

        logger.info("KD Training complete!")
