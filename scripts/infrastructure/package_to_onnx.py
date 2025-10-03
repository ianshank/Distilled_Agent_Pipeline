#!/usr/bin/env python3
"""
CI/CD ONNX Packaging Script
==========================

Packages successful adapters to ONNX format for optimized inference.
"""

import asyncio
import click
import json
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.automated_training_system import (
    AutomatedTrainingSystem,
    InfrastructureConfig
)

@click.command()
@click.option('--role', required=True, help='Agent role')
@click.option('--adapter_path', required=True, help='Path to trained adapter')
@click.option('--output_path', required=True, help='Output path for ONNX model')
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--bucket', default='mangomas-agent-training', help='S3 bucket')
@click.option('--table', default='agent-skill-registry', help='DynamoDB table')
def package_to_onnx(role, adapter_path, output_path, region, bucket, table):
    """Package adapter to ONNX format"""
    
    config = InfrastructureConfig(
        aws_region=region,
        s3_bucket=bucket,
        dynamodb_table=table
    )
    
    result = asyncio.run(run_packaging(config, role, adapter_path, output_path))
    
    if result["status"] == "success":
        print(f"✅ ONNX packaging completed")
        print(f"::set-output name=onnx_path::{result['onnx_model_path']}")
        sys.exit(0)
    else:
        print(f"❌ ONNX packaging failed: {result.get('error', 'Unknown')}")
        sys.exit(1)

async def run_packaging(config, role, adapter_path, output_path):
    """Execute ONNX packaging"""
    system = AutomatedTrainingSystem(config)
    return await system.package_to_onnx(role, adapter_path)

if __name__ == "__main__":
    package_to_onnx() 