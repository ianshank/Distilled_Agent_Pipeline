#!/usr/bin/env python3
"""
MangoMAS Software Development Agent Distillation
Specialized training script for creating a software development assistant
"""

import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from train_distilled_adapter import AgentDistillationTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_software_dev_dataset(input_file: str, output_file: str):
    """Prepare and validate the software development training dataset"""
    logger.info(f"Preparing software development dataset from {input_file}")
    
    # Read and validate the dataset
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    validated_data = []
    for i, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())
            if 'prompt' in data and 'completion' in data:
                # Validate prompt and completion
                if len(data['prompt']) > 10 and len(data['completion']) > 10:
                    validated_data.append(data)
                else:
                    logger.warning(f"Line {i}: Prompt or completion too short")
            else:
                logger.warning(f"Line {i}: Missing prompt or completion field")
        except json.JSONDecodeError:
            logger.warning(f"Line {i}: Invalid JSON format")
    
    # Write validated data
    with open(output_file, 'w') as f:
        for data in validated_data:
            f.write(json.dumps(data) + '\n')
    
    logger.info(f"Dataset prepared: {len(validated_data)} valid examples")
    return len(validated_data)

def create_software_dev_config():
    """Create optimized configuration for software development agent"""
    return {
        # Model configuration
        'teacher_model_name': 'mistralai/Mistral-7B-v0.1',
        'student_model_name': 'microsoft/DialoGPT-medium',
        
        # Training configuration
        'num_train_epochs': 5,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'gradient_accumulation_steps': 4,
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'warmup_steps': 50,
        
        # Distillation configuration
        'distillation_alpha': 0.6,  # Higher weight for knowledge distillation
        'temperature': 2.5,  # Higher temperature for softer knowledge transfer
        'max_length': 1024,  # Longer sequences for code generation
        
        # LoRA configuration
        'use_lora': True,
        'lora_r': 32,  # Higher rank for more capacity
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        'lora_target_modules': 'q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
        
        # Optimization
        'use_fp16': True,
        'use_device_map': True,
        
        # Logging
        'logging_steps': 50,
        'save_steps': 200,
        'save_total_limit': 3,
        'eval_steps': 200,
        'use_wandb': True
    }

def main():
    """Main training function for software development agent"""
    parser = argparse.ArgumentParser(description="Train Software Development Agent")
    parser.add_argument('--input_file', type=str, default='sample_training_data.jsonl',
                       help='Input training data file')
    parser.add_argument('--output_dir', type=str, default='./software_dev_agent',
                       help='Output directory for trained model')
    parser.add_argument('--teacher_model', type=str, default='mistralai/Mistral-7B-v0.1',
                       help='Teacher model for distillation')
    parser.add_argument('--student_model', type=str, default='microsoft/DialoGPT-medium',
                       help='Student model for distillation')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--distillation_alpha', type=float, default=0.6,
                       help='Weight for distillation loss')
    parser.add_argument('--temperature', type=float, default=2.5,
                       help='Temperature for knowledge distillation')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for tracking')
    parser.add_argument('--prepare_data_only', action='store_true',
                       help='Only prepare dataset, skip training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare dataset
    input_file = os.path.join('training', args.input_file)
    prepared_file = os.path.join(args.output_dir, 'prepared_data.jsonl')
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
    
    num_examples = prepare_software_dev_dataset(input_file, prepared_file)
    
    if args.prepare_data_only:
        logger.info("Dataset preparation completed. Skipping training.")
        return
    
    if num_examples < 10:
        logger.warning(f"Only {num_examples} examples available. Consider adding more training data.")
    
    # Create training configuration
    config = create_software_dev_config()
    
    # Override with command line arguments
    config.update({
        'teacher_model_name': args.teacher_model,
        'student_model_name': args.student_model,
        'num_train_epochs': args.epochs,
        'per_device_train_batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'distillation_alpha': args.distillation_alpha,
        'temperature': args.temperature,
        'use_wandb': args.use_wandb,
        'train_file': prepared_file,
        'output_dir': args.output_dir
    })
    
    # Convert config to namespace
    from types import SimpleNamespace
    training_args = SimpleNamespace(**config)
    
    # Initialize and start training
    logger.info("Starting software development agent training...")
    logger.info(f"Configuration: {config}")
    
    try:
        trainer = AgentDistillationTrainer(training_args)
        trainer.train()
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        
        # Save training metadata
        metadata = {
            "training_config": config,
            "dataset_info": {
                "num_examples": num_examples,
                "input_file": input_file,
                "prepared_file": prepared_file
            },
            "training_stats": {
                "completed_at": datetime.now().isoformat(),
                "teacher_model": args.teacher_model,
                "student_model": args.student_model,
                "distillation_alpha": args.distillation_alpha,
                "temperature": args.temperature
            }
        }
        
        metadata_file = os.path.join(args.output_dir, "training_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to: {metadata_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 