#!/usr/bin/env python3
"""
CI/CD Agent Evaluation Script
============================

Evaluates trained agent skills using comprehensive test suites
and integrates with GitHub Actions workflow.
"""

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
from rich.table import Table

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.automated_training_system import (
    AutomatedTrainingSystem,
    InfrastructureConfig
)

console = Console()
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging for CI/CD environment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training/logs/evaluation.log'),
            logging.StreamHandler()
        ]
    )

@click.command()
@click.option('--role', required=True, help='Agent role to evaluate')
@click.option('--test_suite', required=True, help='Test suite to use')
@click.option('--threshold', default=85, type=float, help='Pass rate threshold')
@click.option('--detailed_report', default=True, type=bool, help='Generate detailed report')
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--bucket', default='mangomas-agent-training', help='S3 bucket')
@click.option('--table', default='agent-skill-registry', help='DynamoDB table')
def evaluate_agent(role, test_suite, threshold, detailed_report, region, bucket, table):
    """Evaluate an agent skill using comprehensive test suite"""
    
    # Setup logging
    os.makedirs('training/logs', exist_ok=True)
    setup_logging()
    
    console.print(f"🧪 [bold blue]Evaluating {role} Agent[/bold blue]")
    console.print(f"📋 Test Suite: {test_suite}")
    console.print(f"🎯 Threshold: {threshold}%")
    
    # Run evaluation
    result = asyncio.run(run_evaluation(
        role=role,
        test_suite=test_suite,
        threshold=threshold,
        detailed_report=detailed_report,
        region=region,
        bucket=bucket,
        table=table
    ))
    
    # Display results
    display_evaluation_results(result)
    
    # Output results for GitHub Actions
    print(f"::set-output name=pass_rate::{result.pass_rate}")
    print(f"::set-output name=passed_tests::{result.passed_tests}")
    print(f"::set-output name=total_tests::{result.total_tests}")
    print(f"::set-output name=evaluation_passed::{result.pass_rate >= threshold}")
    
    # Save evaluation report
    report_file = f"training/reports/evaluation_{role.replace(' ', '_').lower()}.json"
    os.makedirs('training/reports', exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(result.dict(), f, indent=2)
    
    # Generate HTML report if requested
    if detailed_report:
        html_report = generate_html_report(result, role, test_suite)
        html_file = f"training/reports/evaluation_{role.replace(' ', '_').lower()}.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        console.print(f"📄 [blue]Detailed report saved to: {html_file}[/blue]")
    
    # Exit with appropriate code
    if result.pass_rate >= threshold:
        console.print(f"✅ [bold green]Evaluation PASSED ({result.pass_rate:.1f}% >= {threshold}%)[/bold green]")
        sys.exit(0)
    else:
        console.print(f"❌ [bold red]Evaluation FAILED ({result.pass_rate:.1f}% < {threshold}%)[/bold red]")
        sys.exit(1)

async def run_evaluation(role, test_suite, threshold, detailed_report, region, bucket, table):
    """Execute the evaluation workflow"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Initialize evaluation system
        task = progress.add_task("🔧 Initializing evaluation system...", total=None)
        
        config = InfrastructureConfig(
            aws_region=region,
            s3_bucket=bucket,
            dynamodb_table=table
        )
        
        system = AutomatedTrainingSystem(config)
        progress.update(task, description="✅ Evaluation system initialized")
        
        # Find the trained adapter
        progress.update(task, description="🔍 Locating trained adapter...")
        adapter_uri = await find_latest_adapter(system, role)
        
        if not adapter_uri:
            progress.update(task, description="❌ No trained adapter found", completed=True)
            raise ValueError(f"No trained adapter found for role: {role}")
        
        progress.update(task, description=f"📦 Found adapter: {adapter_uri}")
        
        # Execute evaluation
        progress.update(task, description=f"🧪 Running {test_suite} test suite...")
        
        evaluation_result = await system.evaluate_agent_skill(role, adapter_uri)
        
        progress.update(task, description="✅ Evaluation completed!", completed=True)
        
        # Log evaluation details
        logger.info(f"Evaluation completed for {role}")
        logger.info(f"  - Pass rate: {evaluation_result.pass_rate:.1f}%")
        logger.info(f"  - Tests passed: {evaluation_result.passed_tests}/{evaluation_result.total_tests}")
        logger.info(f"  - Threshold: {threshold}%")
        logger.info(f"  - Result: {'PASS' if evaluation_result.pass_rate >= threshold else 'FAIL'}")
        
        return evaluation_result

async def find_latest_adapter(system, role):
    """Find the latest trained adapter for the given role"""
    try:
        # In a real implementation, this would query S3 or the registry
        # For now, construct the expected path
        adapter_path = f"s3://{system.s3_bucket}/adapters/{role.replace(' ', '-').lower()}/model.tar.gz"
        
        # Check if adapter exists (simplified check)
        return adapter_path
    except Exception as e:
        logger.error(f"Failed to find adapter for {role}: {e}")
        return None

def display_evaluation_results(result):
    """Display evaluation results in a formatted table"""
    
    # Create summary table
    table = Table(title=f"📊 Evaluation Results: {result.role}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Status", style="green")
    
    # Add summary rows
    table.add_row("Pass Rate", f"{result.pass_rate:.1f}%", 
                  "✅ PASS" if result.pass_rate >= 85 else "❌ FAIL")
    table.add_row("Tests Passed", f"{result.passed_tests}", "")
    table.add_row("Total Tests", f"{result.total_tests}", "")
    table.add_row("Failed Tests", f"{result.failed_tests}", "")
    
    console.print(table)
    
    # Display individual test results if available
    if "test_results" in result.evaluation_details:
        console.print("\n📝 [bold]Individual Test Results:[/bold]")
        
        for test_result in result.evaluation_details["test_results"]:
            status = "✅" if test_result.get("passed", False) else "❌"
            test_id = test_result.get("test_id", "Unknown")
            score = test_result.get("score", 0)
            
            console.print(f"  {status} {test_id}: {score:.1%}")

def generate_html_report(result, role, test_suite):
    """Generate an HTML evaluation report"""
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Agent Evaluation Report - {role}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                  background: #e6f3ff; border-radius: 5px; }}
        .pass {{ background: #e6ffe6; }}
        .fail {{ background: #ffe6e6; }}
        .test-result {{ margin: 10px 0; padding: 10px; 
                       border-left: 4px solid #ddd; }}
        .test-pass {{ border-color: #4caf50; }}
        .test-fail {{ border-color: #f44336; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Agent Evaluation Report</h1>
        <h2>{role}</h2>
        <p><strong>Test Suite:</strong> {test_suite}</p>
        <p><strong>Evaluation Date:</strong> {result.timestamp}</p>
    </div>
    
    <h3>📊 Summary Metrics</h3>
    <div class="metric {'pass' if result.pass_rate >= 85 else 'fail'}">
        <strong>Pass Rate:</strong> {result.pass_rate:.1f}%
    </div>
    <div class="metric">
        <strong>Tests Passed:</strong> {result.passed_tests}
    </div>
    <div class="metric">
        <strong>Total Tests:</strong> {result.total_tests}
    </div>
    <div class="metric">
        <strong>Failed Tests:</strong> {result.failed_tests}
    </div>
    
    <h3>📝 Individual Test Results</h3>
    """
    
    # Add individual test results
    if "test_results" in result.evaluation_details:
        for test_result in result.evaluation_details["test_results"]:
            passed = test_result.get("passed", False)
            test_id = test_result.get("test_id", "Unknown")
            score = test_result.get("score", 0)
            prompt = test_result.get("prompt", "")
            
            status_class = "test-pass" if passed else "test-fail"
            status_emoji = "✅" if passed else "❌"
            
            html_template += f"""
    <div class="test-result {status_class}">
        <h4>{status_emoji} {test_id} - Score: {score:.1%}</h4>
        <p><strong>Prompt:</strong> {prompt}</p>
    </div>
            """
    
    html_template += """
    <h3>🔧 Technical Details</h3>
    <pre>""" + json.dumps(result.evaluation_details, indent=2) + """</pre>
</body>
</html>
    """
    
    return html_template

if __name__ == "__main__":
    evaluate_agent() 