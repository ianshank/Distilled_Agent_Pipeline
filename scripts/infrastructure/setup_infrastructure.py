#!/usr/bin/env python3
"""
CI/CD Infrastructure Setup Script
================================

Sets up AWS infrastructure required for agent training pipeline.
"""

import asyncio
import click
import json
import logging
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.automated_training_system import (
    AutomatedTrainingSystem,
    InfrastructureConfig
)

@click.command()
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--bucket', default='mangomas-agent-training', help='S3 bucket')
@click.option('--table', default='agent-skill-registry', help='DynamoDB table')
def setup_infrastructure(region, bucket, table):
    """Setup AWS infrastructure for agent training"""
    
    config = InfrastructureConfig(
        aws_region=region,
        s3_bucket=bucket,
        dynamodb_table=table
    )
    
    result = asyncio.run(run_setup(config))
    
    if result["status"] == "completed":
        print("✅ Infrastructure setup completed")
        sys.exit(0)
    else:
        print(f"❌ Infrastructure setup failed: {result.get('error', 'Unknown')}")
        sys.exit(1)

async def run_setup(config):
    """Execute infrastructure setup"""
    system = AutomatedTrainingSystem(config)
    return await system.setup_infrastructure()

if __name__ == "__main__":
    setup_infrastructure() 