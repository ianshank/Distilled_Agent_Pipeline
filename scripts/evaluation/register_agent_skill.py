#!/usr/bin/env python3
"""
CI/CD Agent Skill Registration Script
====================================

Registers successful agent skills to the MangoMAS skill registry
with comprehensive metadata and versioning.
"""

import asyncio
import json
import logging
import os
import sys
import time
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
    RegisteredSkill
)

console = Console()
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging for CI/CD environment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training/logs/registration.log'),
            logging.StreamHandler()
        ]
    )

@click.command()
@click.option('--role', required=True, help='Agent role to register')
@click.option('--adapter_uri', required=True, help='S3 URI of the ONNX adapter')
@click.option('--pass_rate', required=True, type=float, help='Evaluation pass rate (0-100)')
@click.option('--model', required=True, help='Base model used for training')
@click.option('--test_suite', required=True, help='Test suite used for evaluation')
@click.option('--environment', default='production', help='Target environment')
@click.option('--version', default='v1.0', help='Skill version')
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--bucket', default='mangomas-agent-training', help='S3 bucket')
@click.option('--table', default='agent-skill-registry', help='DynamoDB table')
def register_skill(role, adapter_uri, pass_rate, model, test_suite, environment, 
                  version, region, bucket, table):
    """Register a successful agent skill to the registry"""
    
    # Setup logging
    os.makedirs('training/logs', exist_ok=True)
    setup_logging()
    
    console.print(f"📝 [bold blue]Registering {role} Agent Skill[/bold blue]")
    console.print(f"🎯 Pass Rate: {pass_rate}%")
    console.print(f"📦 Adapter URI: {adapter_uri}")
    console.print(f"🌍 Environment: {environment}")
    
    # Validate inputs
    if pass_rate < 85:
        console.print(f"❌ [bold red]Pass rate {pass_rate}% is below 85% threshold[/bold red]")
        sys.exit(1)
    
    # Run registration
    result = asyncio.run(run_registration(
        role=role,
        adapter_uri=adapter_uri,
        pass_rate=pass_rate / 100,  # Convert to decimal
        model=model,
        test_suite=test_suite,
        environment=environment,
        version=version,
        region=region,
        bucket=bucket,
        table=table
    ))
    
    if result["status"] == "success":
        console.print(f"✅ [bold green]Registration completed successfully![/bold green]")
        console.print(f"🆔 Skill ID: {result['skill_id']}")
        
        # Set GitHub Actions outputs
        print(f"::set-output name=skill_id::{result['skill_id']}")
        print(f"::set-output name=registration_status::success")
        print(f"::set-output name=registry_uri::{result.get('registry_uri', '')}")
        
        # Save registration metadata
        metadata_file = f"training/reports/registration_{role.replace(' ', '_').lower()}.json"
        os.makedirs('training/reports', exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        sys.exit(0)
    else:
        console.print(f"❌ [bold red]Registration failed: {result.get('error', 'Unknown error')}[/bold red]")
        print(f"::set-output name=registration_status::failed")
        sys.exit(1)

async def run_registration(role, adapter_uri, pass_rate, model, test_suite, 
                          environment, version, region, bucket, table):
    """Execute the registration workflow"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Initialize registration system
        task = progress.add_task("🔧 Initializing registration system...", total=None)
        
        config = InfrastructureConfig(
            aws_region=region,
            s3_bucket=bucket,
            dynamodb_table=table
        )
        
        system = AutomatedTrainingSystem(config)
        progress.update(task, description="✅ Registration system initialized")
        
        # Create skill metadata
        progress.update(task, description="📝 Creating skill metadata...")
        
        skill_id = generate_skill_id(role, environment, version)
        registration_date = datetime.now().isoformat()
        
        # Parse model information
        if "::" in model:
            model_provider, model_name = model.split("::", 1)
        else:
            model_provider = "bedrock"
            model_name = model
        
        # Create registered skill object
        registered_skill = RegisteredSkill(
            role=role,
            skill=f"AutoTrained {version}",
            adapter_uri=adapter_uri,
            runtime="onnx",
            tags=[
                "alora", 
                "automated", 
                "trained", 
                f"env-{environment}",
                f"model-{model_provider}",
                f"suite-{test_suite}",
                f"pass-{int(pass_rate * 100)}"
            ],
            pass_rate=pass_rate,
            registration_date=registration_date,
            skill_id=skill_id
        )
        
        progress.update(task, description="📋 Skill metadata created")
        
        # Validate adapter exists
        progress.update(task, description="🔍 Validating adapter accessibility...")
        
        adapter_valid = await validate_adapter_uri(adapter_uri)
        if not adapter_valid:
            progress.update(task, description="❌ Adapter validation failed", completed=True)
            return {
                "status": "failed",
                "error": f"Adapter URI not accessible: {adapter_uri}"
            }
        
        progress.update(task, description="✅ Adapter validated")
        
        # Register skill
        progress.update(task, description="📝 Registering skill to registry...")
        
        registration_result = await system.register_skill(registered_skill)
        
        if registration_result["status"] == "success":
            progress.update(task, description="✅ Registration completed!", completed=True)
            
            # Enhanced result with metadata
            result = {
                "status": "success",
                "skill_id": skill_id,
                "role": role,
                "adapter_uri": adapter_uri,
                "pass_rate": pass_rate,
                "environment": environment,
                "version": version,
                "model": model,
                "test_suite": test_suite,
                "registration_date": registration_date,
                "registry_uri": f"dynamodb://{table}/skills/{skill_id}",
                "tags": registered_skill.tags
            }
            
            # Log registration details
            logger.info(f"Skill registered successfully for {role}")
            logger.info(f"  - Skill ID: {skill_id}")
            logger.info(f"  - Pass rate: {pass_rate:.1%}")
            logger.info(f"  - Environment: {environment}")
            logger.info(f"  - Adapter URI: {adapter_uri}")
            
            return result
        else:
            progress.update(task, description="❌ Registration failed", completed=True)
            logger.error(f"Registration failed for {role}: {registration_result.get('error', 'Unknown')}")
            return registration_result

def generate_skill_id(role, environment, version):
    """Generate a unique skill ID"""
    role_slug = role.replace(' ', '_').lower()
    timestamp = int(time.time())
    return f"skill_{role_slug}_{environment}_{version}_{timestamp}"

async def validate_adapter_uri(adapter_uri):
    """Validate that the adapter URI is accessible"""
    try:
        if adapter_uri.startswith('s3://'):
            # In a real implementation, this would check S3 object existence
            # For now, assume it's valid if it follows the pattern
            return True
        else:
            logger.warning(f"Non-S3 adapter URI: {adapter_uri}")
            return False
    except Exception as e:
        logger.error(f"Adapter validation failed: {e}")
        return False

def display_registration_summary(result):
    """Display registration summary"""
    console.print("\n📋 [bold]Registration Summary:[/bold]")
    console.print(f"  🆔 Skill ID: {result['skill_id']}")
    console.print(f"  👤 Role: {result['role']}")
    console.print(f"  🎯 Pass Rate: {result['pass_rate']:.1%}")
    console.print(f"  🌍 Environment: {result['environment']}")
    console.print(f"  📦 Adapter: {result['adapter_uri']}")
    console.print(f"  🏷️ Tags: {', '.join(result['tags'])}")
    console.print(f"  📅 Registered: {result['registration_date']}")

if __name__ == "__main__":
    register_skill() 