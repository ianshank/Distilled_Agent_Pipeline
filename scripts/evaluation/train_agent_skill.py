#!/usr/bin/env python3
"""
CI/CD Agent Training Script
===========================

Integrates with GitHub Actions to train agent skills using the
MangoMAS Automated Training System with ALoRA adapters.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.automated_training_system import (
    AutomatedTrainingSystem,
    InfrastructureConfig,
    AgentTrainingConfig
)

console = Console()
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging for CI/CD environment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training/logs/training.log'),
            logging.StreamHandler()
        ]
    )

@click.command()
@click.option('--role', required=True, help='Agent role to train')
@click.option('--dataset', required=True, help='S3 path to training dataset')
@click.option('--model', required=True, help='Base model (e.g., sagemaker::huggingface/mistral)')
@click.option('--adapter_type', default='ALoRA', help='Adapter type')
@click.option('--instance_type', default='ml.g4dn.large', help='SageMaker instance type')
@click.option('--epochs', default=5, type=int, help='Number of training epochs')
@click.option('--batch_size', default=16, type=int, help='Training batch size')
@click.option('--learning_rate', default=1e-4, type=float, help='Learning rate')
@click.option('--force_retrain', default=False, type=bool, help='Force retrain ignoring cache')
@click.option('--output_dir', required=True, help='S3 output directory')
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--bucket', default='mangomas-agent-training', help='S3 bucket')
@click.option('--table', default='agent-skill-registry', help='DynamoDB table')
def train_agent(role, dataset, model, adapter_type, instance_type, epochs, 
                batch_size, learning_rate, force_retrain, output_dir, 
                region, bucket, table):
    """Train an agent skill using ALoRA adapters"""
    
    # Setup logging
    os.makedirs('training/logs', exist_ok=True)
    setup_logging()
    
    console.print(f"🎯 [bold blue]Training {role} Agent[/bold blue]")
    console.print(f"📦 Dataset: {dataset}")
    console.print(f"🤖 Model: {model}")
    console.print(f"⚙️ Adapter: {adapter_type}")
    console.print(f"🔧 Instance: {instance_type}")
    
    # Run training
    result = asyncio.run(run_training(
        role=role,
        dataset=dataset,
        model=model,
        adapter_type=adapter_type,
        instance_type=instance_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        force_retrain=force_retrain,
        output_dir=output_dir,
        region=region,
        bucket=bucket,
        table=table
    ))
    
    # Output results for GitHub Actions
    if result["status"] == "success":
        console.print(f"✅ [bold green]Training completed successfully![/bold green]")
        
        # Set GitHub Actions outputs
        print(f"::set-output name=adapter_path::{result['model_artifacts']}")
        print(f"::set-output name=training_time::{result.get('training_time', 0)}")
        print(f"::set-output name=instance_type::{result.get('instance_type', instance_type)}")
        print(f"::set-output name=logs::training/logs/training.log")
        
        # Save training metadata
        metadata_file = f"training/reports/training_{role.replace(' ', '_').lower()}.json"
        os.makedirs('training/reports', exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        sys.exit(0)
    else:
        console.print(f"❌ [bold red]Training failed: {result.get('error', 'Unknown error')}[/bold red]")
        sys.exit(1)

async def run_training(role, dataset, model, adapter_type, instance_type, 
                      epochs, batch_size, learning_rate, force_retrain, 
                      output_dir, region, bucket, table):
    """Execute the training workflow"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Initialize training system
        task = progress.add_task("🔧 Initializing training system...", total=None)
        
        config = InfrastructureConfig(
            aws_region=region,
            s3_bucket=bucket,
            dynamodb_table=table
        )
        
        system = AutomatedTrainingSystem(config)
        progress.update(task, description="✅ Training system initialized")
        
        # Parse model configuration
        if "::" in model:
            model_provider, model_name = model.split("::", 1)
        else:
            model_provider = "bedrock"
            model_name = model
        
        # Configure training
        progress.update(task, description="⚙️ Configuring training parameters...")
        
        training_config = AgentTrainingConfig(
            role=role,
            dataset_path=dataset,
            base_model=model_name,
            adapter_type=adapter_type,
            output_path=output_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            evaluation_threshold=0.85,
            use_mock_training=not system.aws_available,
            instance_type=instance_type
        )
        
        # Check if training already exists and not forcing retrain
        if not force_retrain:
            progress.update(task, description="🔍 Checking for existing training...")
            existing = await check_existing_training(system, role, dataset)
            if existing:
                console.print(f"ℹ️ [yellow]Found existing training, using cached result[/yellow]")
                return existing
        
        # Execute training
        progress.update(task, description=f"🎯 Training {role} agent...")
        
        training_result = await system.train_agent_skill(training_config)
        
        if training_result["status"] == "success":
            progress.update(task, description="✅ Training completed successfully!", completed=True)
            
            # Log training details
            logger.info(f"Training completed for {role}")
            logger.info(f"  - Training time: {training_result.get('training_time', 0)}s")
            logger.info(f"  - Model artifacts: {training_result['model_artifacts']}")
            logger.info(f"  - Instance type: {training_result.get('instance_type', instance_type)}")
            
            return training_result
        else:
            progress.update(task, description="❌ Training failed", completed=True)
            logger.error(f"Training failed for {role}: {training_result.get('error', 'Unknown')}")
            return training_result

async def check_existing_training(system, role, dataset):
    """Check if training already exists for this role/dataset combination"""
    try:
        # Simple cache check based on role and dataset hash
        import hashlib
        dataset_hash = hashlib.md5(dataset.encode()).hexdigest()[:8]
        cache_key = f"{role.replace(' ', '_').lower()}_{dataset_hash}"
        
        # In a real implementation, this would check S3 or a cache service
        # For now, return None to always train
        return None
    except Exception as e:
        logger.warning(f"Cache check failed: {e}")
        return None

def validate_arguments(role, dataset, model):
    """Validate training arguments"""
    errors = []
    
    # Validate role
    valid_roles = [
        "Software Engineer", "Quality Engineer", "DevOps Engineer",
        "Security Specialist", "Principal Architect", "Product Manager",
        "VP Software Products"
    ]
    if role not in valid_roles:
        errors.append(f"Invalid role: {role}. Must be one of: {', '.join(valid_roles)}")
    
    # Validate dataset S3 path
    if not dataset.startswith('s3://'):
        errors.append(f"Dataset must be an S3 path starting with 's3://'")
    
    # Validate model format
    if "::" in model:
        provider, model_name = model.split("::", 1)
        if provider not in ["sagemaker", "bedrock"]:
            errors.append(f"Model provider must be 'sagemaker' or 'bedrock', got: {provider}")
    
    if errors:
        for error in errors:
            console.print(f"❌ [bold red]{error}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    train_agent() 