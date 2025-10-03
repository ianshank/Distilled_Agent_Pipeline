#!/usr/bin/env python3
"""
MangoMAS Agent Distillation Training Script
SageMaker-compatible training script for distilling knowledge from large models to smaller, specialized agents
"""

import os
import argparse
import json
import logging
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, Dataset

# Handle peft import compatibility
try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    logger.warning(f"PEFT import failed: {e}. Trying alternative import...")
    try:
        # Try importing from specific modules
        from peft import LoraConfig
        from peft import get_peft_model
        from peft import TaskType
    except ImportError:
        logger.error("PEFT library not available. Please install compatible version.")
        raise

import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentDistillationTrainer:
    """Advanced trainer for agent distillation with knowledge transfer"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_wandb()
        
    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        if self.args.use_wandb:
            wandb.init(
                project="mangomas-agent-distillation",
                name=f"distillation-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(self.args)
            )
    
    def load_teacher_model(self) -> AutoModelForCausalLM:
        """Load the teacher model for knowledge distillation"""
        logger.info(f"Loading teacher model: {self.args.teacher_model_name}")
        
        teacher_model = AutoModelForCausalLM.from_pretrained(
            self.args.teacher_model_name,
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32,
            device_map="auto" if self.args.use_device_map else None,
            trust_remote_code=True
        )
        
        # Freeze teacher model parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        return teacher_model
    
    def load_student_model(self) -> AutoModelForCausalLM:
        """Load the student model for distillation"""
        logger.info(f"Loading student model: {self.args.student_model_name}")
        
        student_model = AutoModelForCausalLM.from_pretrained(
            self.args.student_model_name,
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32,
            device_map="auto" if self.args.use_device_map else None,
            trust_remote_code=True
        )
        
        return student_model
    
    def setup_lora_config(self) -> LoraConfig:
        """Setup LoRA configuration for efficient fine-tuning"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=self.args.lora_target_modules.split(",")
        )
    
    def prepare_dataset(self) -> Dataset:
        """Prepare and preprocess the training dataset"""
        # Use SageMaker input directory structure
        train_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
        
        # Find the actual uploaded file
        import glob
        jsonl_files = glob.glob(os.path.join(train_dir, '*.jsonl'))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {train_dir}")
        
        train_file = jsonl_files[0]  # Use the first .jsonl file found
        logger.info(f"Loading dataset from: {train_file}")
        
        # Load dataset
        if train_file.endswith('.jsonl'):
            dataset = load_dataset('json', data_files={'train': train_file})
        else:
            dataset = load_dataset(train_file)
        
        # Tokenize dataset
        tokenizer = AutoTokenizer.from_pretrained(self.args.student_model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            # Handle different input formats
            if 'prompt' in examples:
                texts = examples['prompt']
            elif 'text' in examples:
                texts = examples['text']
            elif 'input' in examples:
                texts = examples['input']
            else:
                # Assume first column is text
                texts = list(examples.values())[0]
            
            # Tokenize with truncation and padding
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.args.max_length,
                return_tensors='pt'
            )
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        logger.info(f"Dataset prepared: {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def create_distillation_loss(self, student_outputs, teacher_outputs, labels):
        """Compute distillation loss combining task loss and knowledge distillation"""
        # Task loss (standard language modeling)
        task_loss = torch.nn.functional.cross_entropy(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Knowledge distillation loss
        if self.args.distillation_alpha > 0:
            # Temperature scaling
            student_logits = student_outputs.logits / self.args.temperature
            teacher_logits = teacher_outputs.logits / self.args.temperature
            
            # KL divergence loss
            distillation_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            ) * (self.args.temperature ** 2)
            
            # Combined loss
            total_loss = (1 - self.args.distillation_alpha) * task_loss + \
                        self.args.distillation_alpha * distillation_loss
        else:
            total_loss = task_loss
        
        return total_loss
    
    def train(self):
        """Main training loop with distillation"""
        logger.info("Starting agent distillation training")
        
        # Load models
        teacher_model = self.load_teacher_model()
        student_model = self.load_student_model()
        
        # Setup LoRA for student model
        if self.args.use_lora:
            lora_config = self.setup_lora_config()
            student_model = get_peft_model(student_model, lora_config)
            student_model.print_trainable_parameters()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset()
        
        # Setup data collator
        tokenizer = AutoTokenizer.from_pretrained(self.args.student_model_name)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_steps=self.args.warmup_steps,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            evaluation_strategy="steps" if self.args.eval_file else "no",
            eval_steps=self.args.eval_steps if self.args.eval_file else None,
            load_best_model_at_end=True if self.args.eval_file else False,
            metric_for_best_model="eval_loss" if self.args.eval_file else None,
            greater_is_better=False if self.args.eval_file else None,
            fp16=self.args.use_fp16,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if self.args.use_wandb else None,
            logging_dir=f"{self.args.output_dir}/logs",
            run_name=f"distillation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        # Custom trainer with distillation
        trainer = DistillationTrainer(
            model=student_model,
            teacher_model=teacher_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=self.prepare_eval_dataset() if self.args.eval_file else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            distillation_alpha=self.args.distillation_alpha,
            temperature=self.args.temperature
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        final_output_dir = os.path.join(self.args.output_dir, "final")
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        # Save training metadata
        self.save_training_metadata(final_output_dir)
        
        logger.info(f"Training completed. Model saved to: {final_output_dir}")
        
        if self.args.use_wandb:
            wandb.finish()
    
    def prepare_eval_dataset(self) -> Optional[Dataset]:
        """Prepare evaluation dataset if provided"""
        if not self.args.eval_file:
            return None
            
        logger.info(f"Loading evaluation dataset from: {self.args.eval_file}")
        
        if self.args.eval_file.endswith('.jsonl'):
            dataset = load_dataset('json', data_files={'eval': self.args.eval_file})
        else:
            dataset = load_dataset(self.args.eval_file)
        
        tokenizer = AutoTokenizer.from_pretrained(self.args.student_model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            if 'prompt' in examples:
                texts = examples['prompt']
            elif 'text' in examples:
                texts = examples['text']
            elif 'input' in examples:
                texts = examples['input']
            else:
                texts = list(examples.values())[0]
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.args.max_length,
                return_tensors='pt'
            )
            
            return tokenized
        
        tokenized_dataset = dataset['eval'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['eval'].column_names
        )
        
        return tokenized_dataset
    
    def save_training_metadata(self, output_dir: str):
        """Save training metadata and configuration"""
        metadata = {
            "training_config": vars(self.args),
            "model_info": {
                "teacher_model": self.args.teacher_model_name,
                "student_model": self.args.student_model_name,
                "distillation_alpha": self.args.distillation_alpha,
                "temperature": self.args.temperature
            },
            "training_stats": {
                "completed_at": datetime.now().isoformat(),
                "device": str(self.device),
                "use_lora": self.args.use_lora,
                "use_fp16": self.args.use_fp16
            }
        }
        
        metadata_path = os.path.join(output_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to: {metadata_path}")


class DistillationTrainer(Trainer):
    """Custom trainer with knowledge distillation capabilities"""
    
    def __init__(self, teacher_model, distillation_alpha, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distillation_alpha = distillation_alpha
        self.temperature = temperature
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with knowledge distillation"""
        # Get student outputs
        student_outputs = model(**inputs)
        
        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        # Compute distillation loss
        loss = self.create_distillation_loss(
            student_outputs, 
            teacher_outputs, 
            inputs.get("labels")
        )
        
        return (loss, student_outputs) if return_outputs else loss
    
    def create_distillation_loss(self, student_outputs, teacher_outputs, labels):
        """Compute distillation loss combining task loss and knowledge distillation"""
        # Task loss (standard language modeling)
        task_loss = torch.nn.functional.cross_entropy(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Knowledge distillation loss
        if self.distillation_alpha > 0:
            # Temperature scaling
            student_logits = student_outputs.logits / self.temperature
            teacher_logits = teacher_outputs.logits / self.temperature
            
            # KL divergence loss
            distillation_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Combined loss
            total_loss = (1 - self.distillation_alpha) * task_loss + \
                        self.distillation_alpha * distillation_loss
        else:
            total_loss = task_loss
        
        return total_loss


def main():
    """Main function with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(description="MangoMAS Agent Distillation Training")
    
    # Model configuration
    parser.add_argument('--teacher_model_name', type=str, 
                       default='mistralai/Mistral-7B-v0.1',
                       help='Teacher model for knowledge distillation')
    parser.add_argument('--student_model_name', type=str,
                       default='microsoft/DialoGPT-medium',
                       help='Student model to be distilled')
    
    # Data configuration
    parser.add_argument('--train_file', type=str, default='train.jsonl',
                       help='Training data file (JSONL format)')
    parser.add_argument('--eval_file', type=str, default=None,
                       help='Evaluation data file (optional)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Training configuration
    parser.add_argument('--output_dir', type=str, default='/opt/ml/model',
                       help='Output directory for model and logs')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2,
                       help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2,
                       help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    
    # Distillation configuration
    parser.add_argument('--distillation_alpha', type=float, default=0.5,
                       help='Weight for distillation loss (0-1)')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Temperature for knowledge distillation')
    
    # LoRA configuration
    parser.add_argument('--use_lora', type=str, default='True',
                       help='Use LoRA for efficient fine-tuning (True/False)')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str,
                       default='q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
                       help='Comma-separated list of target modules for LoRA')
    
    # Optimization configuration
    parser.add_argument('--use_fp16', type=str, default='False',
                       help='Use FP16 precision (True/False)')
    parser.add_argument('--use_device_map', type=str, default='False',
                       help='Use device map for model loading (True/False)')
    
    # Logging and monitoring
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save frequency')
    parser.add_argument('--save_total_limit', type=int, default=2,
                       help='Maximum number of checkpoints to save')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluation frequency')
    parser.add_argument('--use_wandb', type=str, default='False',
                       help='Use Weights & Biases for experiment tracking (True/False)')
    
    # Agent-specific configuration (for compatibility with orchestration script)
    parser.add_argument('--adapter_type', type=str, default='default_alora',
                       help='Type of adapter for the agent')
    parser.add_argument('--agent_name', type=str, default='default-agent',
                       help='Name of the agent being trained')
    parser.add_argument('--agent_role', type=str, default='Default',
                       help='Role of the agent')
    parser.add_argument('--agent_specialization', type=str, default='Default',
                       help='Specialization of the agent')
    parser.add_argument('--capabilities', type=str, default='',
                       help='Comma-separated list of agent capabilities')
    
    args = parser.parse_args()
    
    # Convert string boolean arguments to actual booleans
    args.use_fp16 = args.use_fp16.lower() == 'true'
    args.use_device_map = args.use_device_map.lower() == 'true'
    args.use_wandb = args.use_wandb.lower() == 'true'
    args.use_lora = args.use_lora.lower() == 'true' if hasattr(args, 'use_lora') else True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer and start training
    trainer = AgentDistillationTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main() 