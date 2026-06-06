"""
Teacher-student fidelity metrics for distillation evaluation.

Unlike the generation-quality metrics in :mod:`nlm.eval.metrics`, these compare
the *distributions* produced by a student against its teacher. They are the
distinctive signal for knowledge distillation: a well-distilled student should
agree with the teacher's next-token predictions and have low divergence from
the teacher's soft targets.

The pure tensor functions (``top1_agreement``, ``kl_fidelity``) are framework
math only and are unit tested with synthetic logits. ``evaluate_fidelity``
drives two loaded models over a set of prompts and is exercised in integration
runs.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def top1_agreement(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Fraction of positions where student and teacher argmax tokens agree.

    Args:
        student_logits: Student logits (B, L, V).
        teacher_logits: Teacher logits (B, L, V).
        attention_mask: Optional mask (B, L); 0 positions are ignored.

    Returns:
        Top-1 agreement rate in [0.0, 1.0].
    """
    # Guard against differing vocab padding between the two models.
    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    student_pred = student_logits[..., :min_vocab].argmax(dim=-1)
    teacher_pred = teacher_logits[..., :min_vocab].argmax(dim=-1)
    match = (student_pred == teacher_pred)

    if attention_mask is not None:
        mask = attention_mask.bool()
        valid = mask.sum().item()
        if valid == 0:
            return 0.0
        return (match & mask).sum().item() / valid

    return match.float().mean().item()


def kl_fidelity(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Mean KL divergence KL(teacher || student) over valid positions.

    Lower is better (0.0 means identical distributions). Temperature scaling
    mirrors the distillation loss in :func:`nlm.training.trainer.compute_distillation_loss`.

    Args:
        student_logits: Student logits (B, L, V).
        teacher_logits: Teacher logits (B, L, V).
        temperature: Softmax temperature.
        attention_mask: Optional mask (B, L); 0 positions are ignored.

    Returns:
        Mean per-position KL divergence (>= 0.0).
    """
    # Guard against differing vocab padding between the two models.
    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    student_logits = student_logits[..., :min_vocab]
    teacher_logits = teacher_logits[..., :min_vocab]

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

    # Per-position KL: sum over vocab of p * (log p - log q). Passing both as
    # log-probs with log_target=True is numerically stable and avoids NaNs when
    # FP16 softmax probabilities underflow to 0.0.
    per_position = F.kl_div(
        student_log_probs, teacher_log_probs, reduction="none", log_target=True
    ).sum(dim=-1) * (temperature ** 2)

    if attention_mask is not None:
        mask = attention_mask.to(per_position.dtype)
        denom = mask.sum()
        if denom.item() == 0:
            return 0.0
        return (per_position * mask).sum().item() / denom.item()

    return per_position.mean().item()


def evaluate_fidelity(
    student_model,
    teacher_model,
    tokenizer,
    prompts: List[str],
    max_length: int = 256,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute aggregate fidelity metrics between a student and teacher model.

    Args:
        student_model: Loaded student model (eval mode).
        teacher_model: Loaded teacher model (eval mode).
        tokenizer: Tokenizer shared/compatible across both models.
        prompts: Prompts to evaluate fidelity over.
        max_length: Max tokenized sequence length.
        temperature: Softmax temperature for KL.
        device: Device to run on (defaults to student model's device).

    Returns:
        Dict with ``top1_agreement`` and ``kl_divergence`` averaged over prompts.
    """
    if device is None:
        device = next(student_model.parameters()).device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student_model.eval()
    teacher_model.eval()

    agreements: List[float] = []
    divergences: List[float] = []

    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = inputs.get("attention_mask")

        with torch.no_grad():
            student_logits = student_model(**inputs).logits
            teacher_logits = teacher_model(**inputs).logits

        agreements.append(top1_agreement(student_logits, teacher_logits, attention_mask))
        divergences.append(
            kl_fidelity(student_logits, teacher_logits, temperature, attention_mask)
        )

    n = max(len(prompts), 1)
    return {
        "top1_agreement": sum(agreements) / n,
        "kl_divergence": sum(divergences) / n,
    }
