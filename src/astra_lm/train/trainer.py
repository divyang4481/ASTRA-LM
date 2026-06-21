import os
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
from ..utils.memory import log_cuda_memory, cleanup_memory, log_nvidia_smi
from ..utils.seed import set_seed

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

        # Set seed early
        set_seed(self.config.seed)

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

        log_cuda_memory("Before model to device")
        self.model = model.to(self.device)
        log_cuda_memory("After model to device")
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
                    # T4 generally should use fp16. A100+ should use bf16.
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

        # Adjust steps for scheduler based on gradient accumulation updates
        scheduler_warmup_steps = max(1, self.config.warmup_steps // self.config.gradient_accumulation_steps)
        scheduler_training_steps = max(1, self.config.max_steps // self.config.gradient_accumulation_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=scheduler_warmup_steps,
            num_training_steps=scheduler_training_steps,
            min_lr_ratio=self.config.min_lr_ratio
        )

        self.global_step = 0
        self.total_tokens_trained = 0

        # Output directory check
        if os.path.exists(self.config.output_dir) and os.listdir(self.config.output_dir) and not self.config.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({self.config.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        # Initialize CSV metrics log file
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.config.output_dir, "metrics.csv")
        # Always write header since we should be in a fresh or overwritten directory
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            header = "step,loss,eval_loss,eval_perplexity,lr,candidate_ratio,attention_candidate_mode,elapsed_time,seed"
            # Add VayuSphere columns
            header += (
                ",vs_q_gate_mean,vs_q_gate_std,vs_q_gate_min,vs_q_gate_max"
                ",vs_k_gate_mean,vs_k_gate_std,vs_k_gate_min,vs_k_gate_max"
                ",vs_centroid_grad_norm,vs_centroid_usage_entropy,vs_top_centroid_usage_ratio"
                ",vs_per_layer_q_gate_mean,vs_per_layer_k_gate_mean\n"
            )
            f.write(header)

    def train(self):
        """
        Runs the training loop.
        """
        logger.info("Starting training...")
        log_nvidia_smi()
        logger.info(f"Total steps: {self.config.max_steps}")
        logger.info(f"Device: {self.device}")

        self.model.train()

        # Ensure dataloader uses seed if it's shufflable
        if hasattr(self.train_dataloader, "generator") and self.train_dataloader.generator is not None:
            self.train_dataloader.generator.manual_seed(self.config.seed)

        # Create an infinite iterator for the training dataloader
        def get_train_batch():
            while True:
                for batch in self.train_dataloader:
                    yield batch

        train_iter = iter(get_train_batch())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time_total = time.time()
        start_time = time.time()
        tokens_processed = 0
        
        start_step = self.global_step + 1
        
        for step in range(start_step, self.config.max_steps + 1):
            self.global_step = step
            batch = next(train_iter)

            input_ids = batch["input_ids"].to(self.device)
            if step == 1:
                log_cuda_memory("After first batch to device")
                log_nvidia_smi()
            labels = batch["labels"].to(self.device) if "labels" in batch else input_ids

            # Only calculate diagnostics on logging steps to save performance
            return_diagnostics = (step % self.config.logging_steps == 0)

            # Forward pass with AMP
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    return_diagnostics=return_diagnostics
                )
                loss = outputs["loss"]
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            if step == 1:
                log_cuda_memory("After first forward")
                log_nvidia_smi()

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Collect centroid gradients if VayuSphere is active and it's a logging step
            if return_diagnostics and self.model.config.use_vayusphere:
                centroid_grads = []
                for name, param in self.model.named_parameters():
                    if "centroids" in name and param.grad is not None:
                        centroid_grads.append(param.grad.detach().norm().item())

                if centroid_grads:
                    outputs["vayusphere_centroid_grad_norm_mean"] = sum(centroid_grads) / len(centroid_grads)
                    outputs["vayusphere_centroid_grad_norm_max"] = max(centroid_grads)

            if step == 1:
                log_cuda_memory("After first backward")
                log_nvidia_smi()

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

                if step == self.config.gradient_accumulation_steps:
                    log_cuda_memory("After first optimizer step")
                    log_nvidia_smi()

            # Track tokens
            batch_size, seq_len = input_ids.shape
            tokens_processed += batch_size * seq_len
            self.total_tokens_trained += batch_size * seq_len

            # Logging
            if step % self.config.logging_steps == 0:
                if step % (self.config.logging_steps * 10) == 0:
                    log_nvidia_smi()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                current_lr = self.scheduler.get_last_lr()[0]

                # Aggregate candidate ratios across all layers
                candidate_ratios = []
                candidate_mode = "none"
                if "diagnostics" in outputs:
                    candidate_ratios = [
                        diag["candidate_ratio"] 
                        for diag in outputs["diagnostics"] 
                        if diag and "candidate_ratio" in diag
                    ]
                    # Determine candidate mode
                    if any("chakra" in str(diag.get("mode", "")) for diag in outputs["diagnostics"] if diag):
                         candidate_mode = "chakra_legacy"
                    elif self.model.config.use_vayusphere and self.model.config.vayusphere_alpha > 0:
                        # For now qk_gate has no pruning, so it's none, but we might have vayusphere_sparse later
                        pass

                diag_str = ""
                avg_ratio_str = ""
                if candidate_ratios:
                    avg_ratio = sum(candidate_ratios) / len(candidate_ratios)
                    avg_ratio_str = f"{avg_ratio:.4f}"
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

                # Write training metrics to CSV
                step_loss = loss.item() * self.config.gradient_accumulation_steps

                # Extract VayuSphere diagnostics
                vs_diag = {
                    "q_gate_mean": "", "q_gate_std": "", "q_gate_min": "", "q_gate_max": "",
                    "k_gate_mean": "", "k_gate_std": "", "k_gate_min": "", "k_gate_max": "",
                    "centroid_grad_norm": outputs.get("vayusphere_centroid_grad_norm_mean", ""),
                    "centroid_usage_entropy": "", "top_centroid_usage_ratio": "",
                    "per_layer_q_gate_mean": "", "per_layer_k_gate_mean": ""
                }

                if "diagnostics" in outputs:
                    q_gate_means = [d["vayusphere_q_gate_mean"] for d in outputs["diagnostics"] if d and "vayusphere_q_gate_mean" in d]
                    q_gate_stds = [d["vayusphere_q_gate_std"] for d in outputs["diagnostics"] if d and "vayusphere_q_gate_std" in d]
                    q_gate_mins = [d["vayusphere_q_gate_min"] for d in outputs["diagnostics"] if d and "vayusphere_q_gate_min" in d]
                    q_gate_maxs = [d["vayusphere_q_gate_max"] for d in outputs["diagnostics"] if d and "vayusphere_q_gate_max" in d]
                    
                    k_gate_means = [d["vayusphere_k_gate_mean"] for d in outputs["diagnostics"] if d and "vayusphere_k_gate_mean" in d]
                    k_gate_stds = [d["vayusphere_k_gate_std"] for d in outputs["diagnostics"] if d and "vayusphere_k_gate_std" in d]
                    k_gate_mins = [d["vayusphere_k_gate_min"] for d in outputs["diagnostics"] if d and "vayusphere_k_gate_min" in d]
                    k_gate_maxs = [d["vayusphere_k_gate_max"] for d in outputs["diagnostics"] if d and "vayusphere_k_gate_max" in d]

                    q_entropies = [d["vayusphere_q_centroid_usage_entropy"] for d in outputs["diagnostics"] if d and "vayusphere_q_centroid_usage_entropy" in d]
                    k_entropies = [d["vayusphere_k_centroid_usage_entropy"] for d in outputs["diagnostics"] if d and "vayusphere_k_centroid_usage_entropy" in d]
                    
                    q_top_ratios = [d["vayusphere_q_top_centroid_usage_ratio"] for d in outputs["diagnostics"] if d and "vayusphere_q_top_centroid_usage_ratio" in d]
                    k_top_ratios = [d["vayusphere_k_top_centroid_usage_ratio"] for d in outputs["diagnostics"] if d and "vayusphere_k_top_centroid_usage_ratio" in d]

                    if q_gate_means:
                        vs_diag["q_gate_mean"] = sum(q_gate_means) / len(q_gate_means)
                        vs_diag["q_gate_std"] = sum(q_gate_stds) / len(q_gate_stds) if q_gate_stds else 0.0
                        vs_diag["q_gate_min"] = min(q_gate_mins)
                        vs_diag["q_gate_max"] = max(q_gate_maxs)
                        vs_diag["per_layer_q_gate_mean"] = ";".join([f"{m:.4f}" for m in q_gate_means])
                    if k_gate_means:
                        vs_diag["k_gate_mean"] = sum(k_gate_means) / len(k_gate_means)
                        vs_diag["k_gate_std"] = sum(k_gate_stds) / len(k_gate_stds) if k_gate_stds else 0.0
                        vs_diag["k_gate_min"] = min(k_gate_mins)
                        vs_diag["k_gate_max"] = max(k_gate_maxs)
                        vs_diag["per_layer_k_gate_mean"] = ";".join([f"{m:.4f}" for m in k_gate_means])

                    entropies = q_entropies + k_entropies
                    if entropies:
                        vs_diag["centroid_usage_entropy"] = sum(entropies) / len(entropies)
                    top_ratios = q_top_ratios + k_top_ratios
                    if top_ratios:
                        vs_diag["top_centroid_usage_ratio"] = sum(top_ratios) / len(top_ratios)

                with open(self.metrics_file, "a", encoding="utf-8") as f:
                    vs_str = (
                        f"{vs_diag['q_gate_mean']},{vs_diag['q_gate_std']},{vs_diag['q_gate_min']},{vs_diag['q_gate_max']},"
                        f"{vs_diag['k_gate_mean']},{vs_diag['k_gate_std']},{vs_diag['k_gate_min']},{vs_diag['k_gate_max']},"
                        f"{vs_diag['centroid_grad_norm']},{vs_diag['centroid_usage_entropy']},{vs_diag['top_centroid_usage_ratio']},"
                        f"\"{vs_diag['per_layer_q_gate_mean']}\",\"{vs_diag['per_layer_k_gate_mean']}\""
                    )
                    f.write(
                        f"{step},{step_loss:.4f},,,{current_lr:.2e},{avg_ratio_str},{candidate_mode},"
                        f"{time.time() - self.start_time_total:.2f},{self.config.seed},{vs_str}\n"
                    )

                # Reset tracking for next log interval
                start_time = time.time()
                tokens_processed = 0

            # Evaluation
            if step % self.config.eval_steps == 0 and self.eval_dataloader is not None:
                eval_results = self._evaluate()
                with open(self.metrics_file, "a", encoding="utf-8") as f:
                    # Fill with empty for VS diags during eval
                    vs_empty = "," * 12
                    f.write(f"{step},,{eval_results['loss']:.4f},{eval_results['perplexity']:.4f},,,,{time.time() - self.start_time_total:.2f},{self.config.seed},{vs_empty}\n")

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()

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
        return eval_results
