"""
Model loading and LoRA adapter setup for knowledge distillation.

Provides teacher/student loaders with consistent dtype/device handling (shared
with the inference server) and a guarded LoRA setup that auto-detects valid
target modules across architectures and degrades gracefully when PEFT is
unavailable or no valid targets exist.
"""

import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when peft is absent
    PEFT_AVAILABLE = False

# Leaf module names commonly adapted by LoRA across architectures. Detection is
# allow-listed to these so we never target arbitrary Linear layers (e.g. lm_head)
# that would waste parameters or destabilize training.
_LORA_CANDIDATE_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # llama / mistral / granite
    "gate_proj",
    "up_proj",
    "down_proj",
    "c_attn",
    "c_proj",
    "c_fc",  # gpt2 (Conv1D)
    "query_key_value",
    "dense",  # bloom / falcon
    "wqkv",
    "wo",
    "w1",
    "w2",
    "w3",  # misc MoE / mixtral
]


def _linear_module_types() -> tuple:
    """Return the module classes treated as linear projections (incl. GPT-2 Conv1D)."""
    types: list = [torch.nn.Linear]
    try:
        from transformers.pytorch_utils import Conv1D

        types.append(Conv1D)
    except Exception:  # pragma: no cover - older/newer transformers without Conv1D
        pass
    return tuple(types)


def detect_lora_target_modules(model) -> List[str]:
    """
    Auto-detect LoRA-suitable target module names present in a model.

    Scans for linear/Conv1D leaf modules whose names are in the known
    candidate allow-list. Returns an empty list (never raises) for models with
    no recognized targets, so callers can fall back safely.

    Args:
        model: A torch module (typically a HuggingFace causal LM).

    Returns:
        Sorted list of unique target module leaf names.
    """
    linear_types = _linear_module_types()
    found = set()

    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            leaf = name.split(".")[-1]
            if leaf in _LORA_CANDIDATE_MODULES:
                found.add(leaf)

    targets = sorted(found)
    logger.debug("Detected LoRA target modules: %s", targets)
    return targets


def _existing_leaf_names(model) -> set:
    """Set of all leaf module names present in the model."""
    return {name.split(".")[-1] for name, _ in model.named_modules() if name}


def setup_lora_adapter(
    model: PreTrainedModel,
    rank: int,
    alpha: int,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> PreTrainedModel:
    """
    Wrap a model with a LoRA adapter, guarding against invalid configurations.

    Behavior:
        * If PEFT is not installed, returns the original model unchanged.
        * If ``target_modules`` is None, targets are auto-detected.
        * If no requested target actually exists in the model (or none are
          detected), returns the ORIGINAL model unchanged (caller can train
          without LoRA) rather than raising.

    Args:
        model: Student model to adapt.
        rank: LoRA rank (r).
        alpha: LoRA alpha scaling.
        dropout: LoRA dropout.
        target_modules: Explicit target module names, or None to auto-detect.

    Returns:
        A PEFT-wrapped model, or the original model when LoRA cannot be applied.
    """
    if not PEFT_AVAILABLE:
        logger.warning("peft is not installed; continuing without LoRA")
        return model

    if target_modules is None:
        target_modules = detect_lora_target_modules(model)
        logger.info("Auto-detected LoRA target modules: %s", target_modules)

    valid_targets = [t for t in target_modules if t in _existing_leaf_names(model)]
    if not valid_targets:
        logger.warning(
            "No valid LoRA target modules found (requested=%s); training without LoRA",
            target_modules,
        )
        return model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=valid_targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    try:
        peft_model = get_peft_model(model, lora_config)
    except Exception as e:  # pragma: no cover - depends on peft/model internals
        logger.warning("LoRA setup failed (%s); training without LoRA", e)
        return model

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    pct = (trainable / total * 100) if total else 0.0
    logger.info(
        "LoRA enabled on %s: %s trainable / %s total params (%.3f%%)",
        valid_targets,
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )
    return peft_model


def _resolve_dtype(device: torch.device, use_fp16: bool) -> torch.dtype:
    """FP16 only on CUDA; otherwise FP32 (MPS/CPU FP16 is unstable for training)."""
    if use_fp16 and device.type == "cuda":
        return torch.float16
    return torch.float32


def _load_model(
    model_id: str,
    device: torch.device,
    use_fp16: bool,
    use_device_map: bool,
    freeze: bool,
) -> PreTrainedModel:
    """Shared loader for teacher/student models."""
    dtype = _resolve_dtype(device, use_fp16)
    use_map = bool(use_device_map and device.type == "cuda")

    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if use_map:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    # device_map="auto" already places the model; otherwise move it explicitly.
    if not use_map:
        model = model.to(device)

    role = "teacher" if freeze else "student"
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded %s model '%s': %s params, dtype=%s, device=%s",
        role,
        model_id,
        f"{param_count:,}",
        dtype,
        device,
    )
    return model


def load_teacher_model(
    model_id: str,
    device: torch.device,
    use_fp16: bool = False,
    use_device_map: bool = False,
) -> PreTrainedModel:
    """Load a frozen teacher model in eval mode."""
    return _load_model(model_id, device, use_fp16, use_device_map, freeze=True)


def load_student_model(
    model_id: str,
    device: torch.device,
    use_fp16: bool = False,
    use_device_map: bool = False,
) -> PreTrainedModel:
    """Load a trainable student model."""
    return _load_model(model_id, device, use_fp16, use_device_map, freeze=False)
