import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import logging
import time

from .trainer import Trainer
from .config import TrainConfig
from ..distill.kd_losses import kl_distillation_loss, kl_topk_distillation_loss
from ..utils.memory import log_cuda_memory, log_nvidia_smi

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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        topk_logits: int = 100
    ):
        super().__init__(
            model=student_model,
            train_config=train_config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            device=device
        )
        log_cuda_memory("Before teacher model to device")
        log_nvidia_smi()
        # Quantized models (8-bit or 4-bit) loaded with device_map do not support `.to()`
        is_quantized = False
        if hasattr(teacher_model, "config") and getattr(teacher_model.config, "quantization_config", None) is not None:
            is_quantized = True
            
        if is_quantized:
            logger.info("Teacher model is quantized. Skipping `.to(device)` call.")
            self.teacher_model = teacher_model.eval()
        else:
            try:
                self.teacher_model = teacher_model.to(self.device).eval()
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not move teacher model to device using `.to()`: {e}. Using the model as-is.")
                self.teacher_model = teacher_model.eval()
        log_cuda_memory("After teacher model to device")
        log_nvidia_smi()
        self.alpha = alpha
        self.temperature = temperature
        self.topk_logits = topk_logits

    def train(self):
        """
        Modified training loop with KD and CHAKRA routing logs.
        """
        logger.info("Starting KD training...")
        logger.info(f"Alpha: {self.alpha}, Temperature: {self.temperature}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        self.model.train()
        
        def get_train_batch():
            while True:
                for batch in self.train_dataloader:
                    yield batch

        train_iter = iter(get_train_batch())
        start_time = time.time()
        self.start_time_total = time.time()
        tokens_processed = 0

        for step in range(1, self.config.max_steps + 1):
            self.global_step = step
            batch = next(train_iter)

            input_ids = batch["input_ids"].to(self.device)
            if step == 1:
                log_cuda_memory("After first KD batch to device")
                log_nvidia_smi()
            labels = batch["labels"].to(self.device) if "labels" in batch else input_ids

            # Only calculate diagnostics on logging steps to save performance
            return_diagnostics = (step % self.config.logging_steps == 0)

            # Forward pass with AMP
            try:
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp):
                    # 1. Teacher forward (no grad)
                    # Support teacher on different device (e.g. CPU)
                    teacher_input_ids = input_ids.to(next(self.teacher_model.parameters()).device)

                    with torch.no_grad():
                        if step == 1:
                            log_cuda_memory("Before teacher forward")
                            log_nvidia_smi()
                        teacher_outputs = self.teacher_model(input_ids=teacher_input_ids)
                        teacher_logits = teacher_outputs["logits"]
                        
                        # Top-k compression
                        k = min(self.topk_logits, teacher_logits.size(-1))
                        teacher_topk_values, teacher_topk_indices = torch.topk(teacher_logits, k=k, dim=-1)
                        
                        # Transfer top-k to student device if needed
                        teacher_topk_values = teacher_topk_values.to(self.device)
                        teacher_topk_indices = teacher_topk_indices.to(self.device)

                        # Proactively delete the massive full-vocabulary logits tensor to free VRAM
                        del teacher_logits
                        del teacher_outputs
                        if step == 1:
                            log_cuda_memory("After teacher forward")
                            log_nvidia_smi()

                    # 2. Student forward
                    if step == 1:
                        log_cuda_memory("Before student forward")
                        log_nvidia_smi()
                    student_outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        return_diagnostics=return_diagnostics
                    )
                    student_logits = student_outputs["logits"]
                    ce_loss = student_outputs["loss"]
                    if step == 1:
                        log_cuda_memory("After student forward")
                        log_nvidia_smi()

                    kd_loss = kl_topk_distillation_loss(
                        student_logits=student_logits,
                        teacher_topk_indices=teacher_topk_indices,
                        teacher_topk_values=teacher_topk_values,
                        temperature=self.temperature
                    )

                    # Combined loss
                    loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
                    if step == 1:
                        log_cuda_memory("After loss calculation")
                        log_nvidia_smi()
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                if step == 1:
                    log_cuda_memory("Before backward")
                    log_nvidia_smi()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if step == 1:
                    log_cuda_memory("After backward")
                    log_nvidia_smi()
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    logger.error(torch.cuda.memory_summary())
                raise

            # Optimizer step (only at accumulation boundary)
            if step % self.config.gradient_accumulation_steps == 0:
                if step == self.config.gradient_accumulation_steps:
                    log_cuda_memory("After first optimizer step")
                    log_nvidia_smi()
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
            self.total_tokens_trained += batch_size * seq_len

            # Logging
            if step % self.config.logging_steps == 0:
                if step % (self.config.logging_steps * 10) == 0:
                    log_nvidia_smi()
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
                
                # Write training metrics to CSV
                step_loss = loss.item() * self.config.gradient_accumulation_steps
                avg_ratio = sum(candidate_ratios) / len(candidate_ratios) if candidate_ratios else 1.0
                with open(self.metrics_file, "a", encoding="utf-8") as f:
                    f.write(f"{step},{step_loss:.4f},,,{current_lr:.2e},{avg_ratio:.4f},{time.time() - self.start_time_total:.2f}\n")

                start_time = time.time()
                tokens_processed = 0

            # Evaluation
            if step % self.config.eval_steps == 0 and self.eval_dataloader is not None:
                eval_results = self._evaluate()
                with open(self.metrics_file, "a", encoding="utf-8") as f:
                    f.write(f"{step},,{eval_results['loss']:.4f},{eval_results['perplexity']:.4f},,,{time.time() - self.start_time_total:.2f}\n")

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
