"""
Command-line interface for NLM distillation training.

Provides argparse-based CLI with config file support and proper
boolean flags (no string parsing).
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nlm.config import load_config, TrainingConfig
from nlm.data import load_distillation_dataset
from nlm.models import (
    select_device,
    load_teacher_model,
    load_student_model,
    setup_lora_adapter
)
from nlm.training import DistillationTrainer

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    ]
)
logger = logging.getLogger(__name__)


def setup_wandb(config: TrainingConfig) -> None:
    """
    Initialize Weights & Biases tracking if enabled.
    
    Args:
        config: Training configuration
    """
    if not config.use_wandb:
        return
    
    try:
        import wandb
        wandb.init(
            project="nlm-distillation",
            name=f"{config.agent_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.redacted_dict()
        )
        logger.info("Weights & Biases tracking enabled")
    except ImportError:
        logger.warning("wandb not installed, skipping W&B tracking")
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")


def save_training_metadata(config: TrainingConfig, output_dir: str) -> None:
    """
    Save training metadata and configuration to output directory.
    
    Args:
        config: Training configuration
        output_dir: Directory to save metadata
    """
    metadata = {
        "config": config.redacted_dict(),
        "training_completed_at": datetime.now().isoformat(),
        "device": str(select_device(config.device_preference)),
        "framework_versions": {
            "torch": torch.__version__,
            "python": sys.version
        }
    }
    
    metadata_path = Path(output_dir) / "training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Training metadata saved to {metadata_path}")


def train(config: TrainingConfig) -> None:
    """
    Execute distillation training with given configuration.
    
    Args:
        config: Training configuration object
    
    Raises:
        RuntimeError: If training fails
    """
    logger.info("Starting NLM distillation training")
    logger.info(f"Agent: {config.agent_name} ({config.agent_role})")
    
    # Select device
    device = select_device(config.device_preference)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.student_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_id)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Load dataset
    logger.info("Loading and tokenizing dataset")
    train_dataset = load_distillation_dataset(
        train_file=config.train_file,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    eval_dataset = None
    if config.eval_file:
        logger.info("Loading evaluation dataset")
        eval_dataset = load_distillation_dataset(
            train_file=config.eval_file,
            tokenizer=tokenizer,
            max_length=config.max_length
        )
    
    # Load teacher model
    teacher_model = load_teacher_model(
        model_id=config.teacher_model_id,
        device=device,
        use_fp16=config.use_fp16,
        use_device_map=config.use_device_map
    )
    
    # Load student model
    student_model = load_student_model(
        model_id=config.student_model_id,
        device=device,
        use_fp16=config.use_fp16,
        use_device_map=config.use_device_map
    )
    
    # Setup LoRA if enabled
    if config.lora.enabled:
        logger.info("Setting up LoRA adapters")
        student_model = setup_lora_adapter(
            model=student_model,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=config.lora.target_modules
        )
    
    # Setup W&B tracking
    setup_wandb(config)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        fp16=config.use_fp16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb" if config.use_wandb else "none",
        logging_dir=str(output_dir / "logs"),
        run_name=f"{config.agent_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Create distillation trainer
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        distillation_alpha=config.distillation.alpha,
        temperature=config.distillation.temperature
    )
    
    # Add early stopping callback if eval dataset present
    if eval_dataset:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
    
    # Train
    logger.info("Starting training loop")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Save final model
    final_output_dir = output_dir / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    logger.info(f"Final model saved to {final_output_dir}")
    
    # Save training metadata
    save_training_metadata(config, str(final_output_dir))
    
    # Cleanup W&B
    if config.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="NLM Distillation Training for Granite-4-MoE and compatible models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    
    # Model configuration
    parser.add_argument("--teacher-model-id", type=str, help="Teacher model ID")
    parser.add_argument("--student-model-id", type=str, help="Student model ID")
    
    # Data paths
    parser.add_argument("--train-file", type=str, help="Training JSONL file")
    parser.add_argument("--eval-file", type=str, help="Evaluation JSONL file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    # Training hyperparameters
    parser.add_argument("--num-train-epochs", type=int, help="Number of epochs")
    parser.add_argument("--per-device-train-batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--max-length", type=int, help="Max sequence length")
    
    # Boolean flags (proper argparse booleans)
    parser.add_argument("--use-fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--no-fp16", action="store_false", dest="use_fp16", help="Disable FP16")
    
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA adapters")
    parser.add_argument("--no-lora", action="store_false", dest="use_lora", help="Disable LoRA")
    
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B tracking")
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb", help="Disable W&B")
    
    # LoRA parameters
    parser.add_argument("--lora-rank", type=int, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    
    # Distillation parameters
    parser.add_argument("--distillation-alpha", type=float, help="Distillation weight")
    parser.add_argument("--temperature", type=float, help="Distillation temperature")
    
    # Agent metadata
    parser.add_argument("--agent-name", type=str, help="Agent identifier")
    parser.add_argument("--agent-role", type=str, help="Agent role")
    
    args = parser.parse_args()
    
    # Build overrides from CLI args
    overrides = {}
    
    # Simple mappings
    simple_mappings = {
        "teacher_model_id": args.teacher_model_id,
        "student_model_id": args.student_model_id,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "agent_name": args.agent_name,
        "agent_role": args.agent_role
    }
    
    for key, value in simple_mappings.items():
        if value is not None:
            overrides[key] = value
    
    # Boolean flags
    if hasattr(args, "use_fp16") and args.use_fp16 is not None:
        overrides["use_fp16"] = args.use_fp16
    
    if hasattr(args, "use_wandb") and args.use_wandb is not None:
        overrides["use_wandb"] = args.use_wandb
    
    # LoRA configuration
    if hasattr(args, "use_lora") and args.use_lora is not None:
        if "lora" not in overrides:
            overrides["lora"] = {}
        overrides["lora"]["enabled"] = args.use_lora
    
    if args.lora_rank is not None:
        if "lora" not in overrides:
            overrides["lora"] = {}
        overrides["lora"]["rank"] = args.lora_rank
    
    if args.lora_alpha is not None:
        if "lora" not in overrides:
            overrides["lora"] = {}
        overrides["lora"]["alpha"] = args.lora_alpha
    
    # Distillation configuration
    if args.distillation_alpha is not None:
        if "distillation" not in overrides:
            overrides["distillation"] = {}
        overrides["distillation"]["alpha"] = args.distillation_alpha
    
    if args.temperature is not None:
        if "distillation" not in overrides:
            overrides["distillation"] = {}
        overrides["distillation"]["temperature"] = args.temperature
    
    # Load configuration
    try:
        config = load_config(config_path=args.config, overrides=overrides)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Execute training
    try:
        train(config)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

