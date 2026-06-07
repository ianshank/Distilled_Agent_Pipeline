"""
Model utilities for distillation: device selection, model loading, LoRA setup.

This module is the single source of truth for device selection and model
loading, reused by the training CLI, inference server, and evaluation harness.
"""

from nlm.models.devices import DEFAULT_DEVICE_PREFERENCE, select_device
from nlm.models.loaders import (
    detect_lora_target_modules,
    load_student_model,
    load_teacher_model,
    setup_lora_adapter,
)

__all__ = [
    "DEFAULT_DEVICE_PREFERENCE",
    "select_device",
    "detect_lora_target_modules",
    "setup_lora_adapter",
    "load_teacher_model",
    "load_student_model",
]
