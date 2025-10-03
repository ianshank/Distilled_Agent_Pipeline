"""
Custom trainer for knowledge distillation.

Extends HuggingFace Trainer with distillation loss combining
task loss (cross-entropy) and KL divergence from teacher.
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from transformers import Trainer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0
) -> Dict[str, torch.Tensor]:
    """
    Compute combined distillation loss.
    
    Loss = (1 - alpha) * task_loss + alpha * distillation_loss
    
    Args:
        student_logits: Student model logits (B, L, V)
        teacher_logits: Teacher model logits (B, L, V)
        labels: Ground truth labels (B, L)
        alpha: Weight for distillation loss (0 = pure task, 1 = pure distillation)
        temperature: Softmax temperature for distillation
    
    Returns:
        Dictionary with total_loss, task_loss, distillation_loss
    """
    # Task loss: standard cross-entropy
    # Use sum reduction to avoid NaN when all labels are -100
    task_loss_sum = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="sum"
    )

    # Count non-ignored tokens
    num_valid_tokens = (labels != -100).sum()

    # Avoid division by zero when all tokens are ignored
    if num_valid_tokens > 0:
        task_loss = task_loss_sum / num_valid_tokens
    else:
        # All tokens ignored - return zero loss with gradient
        task_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    if alpha == 0.0:
        # No distillation, only task loss
        return {
            "total_loss": task_loss,
            "task_loss": task_loss,
            "distillation_loss": torch.tensor(0.0, device=task_loss.device)
        }
    
    # Distillation loss: KL divergence with temperature scaling
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature
    
    # Compute KL divergence
    distillation_loss = F.kl_div(
        F.log_softmax(student_logits_scaled, dim=-1),
        F.softmax(teacher_logits_scaled, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)
    
    # Combined loss
    total_loss = (1.0 - alpha) * task_loss + alpha * distillation_loss
    
    return {
        "total_loss": total_loss,
        "task_loss": task_loss,
        "distillation_loss": distillation_loss
    }


class DistillationTrainer(Trainer):
    """
    Custom Trainer with knowledge distillation from teacher model.
    
    Extends HuggingFace Trainer to compute distillation loss by
    combining student task loss with KL divergence from frozen teacher.
    """
    
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        distillation_alpha: float = 0.5,
        temperature: float = 2.0,
        *args,
        **kwargs
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher_model: Frozen teacher model for distillation
            distillation_alpha: Weight for distillation loss (0-1)
            temperature: Temperature for softmax in distillation
            *args: Additional arguments for Trainer
            **kwargs: Additional keyword arguments for Trainer
        """
        super().__init__(*args, **kwargs)
        
        self.teacher_model = teacher_model
        self.distillation_alpha = distillation_alpha
        self.temperature = temperature
        
        # Ensure teacher is in eval mode and on correct device
        self.teacher_model.eval()
        
        logger.info(
            f"DistillationTrainer initialized: alpha={distillation_alpha}, "
            f"temperature={temperature}"
        )
    
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute distillation loss for training step.
        
        Args:
            model: Student model being trained
            inputs: Input batch with input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs
        
        Returns:
            Loss tensor (and optionally outputs)
        """
        # Get student outputs
        student_outputs = model(**inputs)
        
        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        # Extract labels
        labels = inputs.get("labels")
        if labels is None:
            # Fallback: use input_ids as labels for language modeling
            labels = inputs.get("input_ids")
        
        # Compute distillation loss
        loss_dict = compute_distillation_loss(
            student_logits=student_outputs.logits,
            teacher_logits=teacher_outputs.logits,
            labels=labels,
            alpha=self.distillation_alpha,
            temperature=self.temperature
        )
        
        total_loss = loss_dict["total_loss"]
        
        # Log component losses for monitoring
        self.log({
            "train/task_loss": loss_dict["task_loss"].item(),
            "train/distillation_loss": loss_dict["distillation_loss"].item(),
        })
        
        if return_outputs:
            return total_loss, student_outputs
        else:
            return total_loss
    
    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None
    ):
        """
        Prediction step for evaluation.
        
        Args:
            model: Model to evaluate
            inputs: Input batch
            prediction_loss_only: Whether to return only loss
            ignore_keys: Keys to ignore in outputs
        
        Returns:
            Tuple of (loss, logits, labels)
        """
        # Get student outputs
        with torch.no_grad():
            student_outputs = model(**inputs)
            
            # Get teacher outputs for distillation loss
            teacher_outputs = self.teacher_model(**inputs)
            
            labels = inputs.get("labels")
            if labels is None:
                labels = inputs.get("input_ids")
            
            # Compute distillation loss
            loss_dict = compute_distillation_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                labels=labels,
                alpha=self.distillation_alpha,
                temperature=self.temperature
            )
            
            loss = loss_dict["total_loss"]
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, student_outputs.logits, labels)

