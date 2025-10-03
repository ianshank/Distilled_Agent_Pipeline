"""
Configuration schema and loaders for NLM distillation training.

This module provides Pydantic-based configuration with YAML and environment
variable support, following secure-by-default practices.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

logger = logging.getLogger(__name__)


class LoRAConfig(BaseModel):
    """LoRA configuration for parameter-efficient fine-tuning."""
    
    enabled: bool = Field(default=True, description="Enable LoRA adapters")
    rank: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Target module names for LoRA"
    )

    model_config = ConfigDict(frozen=False)


class DistillationConfig(BaseModel):
    """Distillation loss configuration."""
    
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Distillation weight")
    temperature: float = Field(default=2.0, gt=0.0, description="Softmax temperature")

    model_config = ConfigDict(frozen=False)


class TrainingConfig(BaseSettings):
    """
    Main training configuration with environment variable override support.
    
    Environment variables take precedence over YAML file settings.
    Sensitive values should only be provided via environment, never committed.
    """
    
    # Model configuration
    teacher_model_id: str = Field(
        default="sshleifer/tiny-gpt2",
        description="HuggingFace model ID for teacher"
    )
    student_model_id: str = Field(
        default="distilgpt2",
        description="HuggingFace model ID for student"
    )
    
    # Data paths
    train_file: Optional[str] = Field(
        default=None,
        description="Path to training JSONL file"
    )
    eval_file: Optional[str] = Field(
        default=None,
        description="Path to evaluation JSONL file"
    )
    output_dir: str = Field(
        default="outputs/default",
        description="Output directory for models and logs"
    )
    
    # Training hyperparameters
    num_train_epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    per_device_train_batch_size: int = Field(default=2, ge=1, description="Training batch size")
    per_device_eval_batch_size: int = Field(default=2, ge=1, description="Eval batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation")
    learning_rate: float = Field(default=5e-5, gt=0.0, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay")
    warmup_steps: int = Field(default=100, ge=0, description="Warmup steps")
    max_length: int = Field(default=512, ge=1, le=8192, description="Max sequence length")
    
    # Optimization
    use_fp16: bool = Field(default=False, description="Enable mixed precision (FP16)")
    use_device_map: bool = Field(default=False, description="Enable device_map='auto'")
    device_preference: List[Literal["cuda", "mps", "cpu"]] = Field(
        default_factory=lambda: ["cuda", "mps", "cpu"],
        description="Device preference order"
    )
    
    # LoRA and Distillation
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA configuration")
    distillation: DistillationConfig = Field(
        default_factory=DistillationConfig,
        description="Distillation configuration"
    )
    
    # Logging and monitoring
    logging_steps: int = Field(default=100, ge=1, description="Logging frequency")
    save_steps: int = Field(default=500, ge=1, description="Save checkpoint frequency")
    save_total_limit: int = Field(default=2, ge=1, description="Max checkpoints to keep")
    eval_steps: int = Field(default=500, ge=1, description="Evaluation frequency")
    use_wandb: bool = Field(default=False, description="Enable Weights & Biases logging")
    
    # Agent metadata (optional)
    agent_name: str = Field(default="default-agent", description="Agent identifier")
    agent_role: str = Field(default="Default", description="Agent role")
    
    model_config = SettingsConfigDict(
        env_prefix="NLM_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("output_dir", "train_file", "eval_file")
    @classmethod
    def validate_paths(cls, v: Optional[str]) -> Optional[str]:
        """Validate paths are not absolute system paths outside workspace."""
        if v is None:
            return v
        
        path = Path(v)
        
        # Reject absolute paths that escape to system directories
        if path.is_absolute() and str(path).startswith(("/etc", "/sys", "/proc", "/root")):
            raise ValueError(f"Path {v} points to protected system directory")
        
        return v
    
    def redacted_dict(self) -> dict:
        """
        Return configuration as dict with sensitive fields redacted.
        Safe for logging and display.
        """
        config_dict = self.model_dump()
        
        # Redact any fields that might contain secrets
        # (currently none, but reserved for future HF tokens, etc.)
        return config_dict
    
    def log_config(self) -> None:
        """Log configuration snapshot without sensitive data."""
        logger.info("Training configuration loaded", extra={
            "config": self.redacted_dict()
        })


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None
) -> TrainingConfig:
    """
    Load training configuration from YAML file and environment variables.
    
    Priority (highest to lowest):
    1. CLI overrides (passed as dict)
    2. Environment variables (NLM_* prefix)
    3. YAML file
    4. Defaults
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of CLI override values
    
    Returns:
        TrainingConfig instance with merged settings
    
    Raises:
        FileNotFoundError: If config_path specified but not found
        ValueError: If configuration validation fails
    """
    yaml_config = {}
    
    # Load YAML if provided
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        
        logger.info(f"Loaded configuration from {config_path}")
    
    # Merge YAML with overrides
    merged_config = {**yaml_config, **(overrides or {})}
    
    # Pydantic Settings will automatically merge with environment variables
    config = TrainingConfig(**merged_config)
    
    config.log_config()
    return config

