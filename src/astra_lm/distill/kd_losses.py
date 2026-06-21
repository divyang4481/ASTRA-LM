import torch
import torch.nn.functional as F

def kl_distillation_loss(
    student_logits: torch.Tensor, 
    teacher_logits: torch.Tensor, 
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Standard KL divergence loss for logit distillation.
    """
    p_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL Divergence
    loss = F.kl_div(p_s, p_t, reduction="batchmean") * (temperature ** 2)
    return loss

def kl_topk_distillation_loss(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_values: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    KL divergence loss on top-k teacher logits for VRAM efficiency.
    teacher_topk_values are the actual top-k logits (or probabilities).
    """
    # Gather student logits at teacher top-k positions
    # student_logits: [batch_size, seq_len, vocab_size]
    # teacher_topk_indices: [batch_size, seq_len, k]
    # teacher_topk_values: [batch_size, seq_len, k]
    student_topk_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)
    
    p_s = F.log_softmax(student_topk_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_topk_values / temperature, dim=-1)
    
    loss = F.kl_div(p_s, p_t, reduction="batchmean") * (temperature ** 2)
    return loss

def hidden_state_mse_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    """
    MSE loss for hidden state alignment.
    """
    return F.mse_loss(student_hidden, teacher_hidden)
