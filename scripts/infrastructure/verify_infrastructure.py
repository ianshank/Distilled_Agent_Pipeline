#!/usr/bin/env python3
"""
CI/CD Infrastructure Verification Script
=======================================

Verifies AWS infrastructure is ready for agent training.
"""

import asyncio
import click
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
def verify_infrastructure(region, bucket, table):
    """Verify AWS infrastructure is ready"""
    
    config = InfrastructureConfig(
        aws_region=region,
        s3_bucket=bucket,
        dynamodb_table=table
    )
    
    system = AutomatedTrainingSystem(config)
    
    if system.aws_available:
        print("✅ AWS infrastructure verified")
        sys.exit(0)
    else:
        print("❌ AWS infrastructure not available")
        sys.exit(1)

if __name__ == "__main__":
    verify_infrastructure() 