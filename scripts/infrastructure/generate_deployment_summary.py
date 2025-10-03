#!/usr/bin/env python3
"""
CI/CD Deployment Summary Generator
=================================

Generates comprehensive deployment summary for GitHub Actions reporting.
"""

import click
import json
import boto3
import os
from datetime import datetime

@click.command()
@click.option('--environment', default='production', help='Target environment')
@click.option('--workflow_run_id', required=True, help='GitHub workflow run ID')
@click.option('--output', required=True, help='Output file path')
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--table', default='agent-skill-registry', help='DynamoDB table')
def generate_summary(environment, workflow_run_id, output, region, table):
    """Generate deployment summary"""
    
    try:
        # Query deployment results from reports
        agents_data = collect_agent_results()
        
        # Generate summary
        summary = {
            "workflow_run_id": workflow_run_id,
            "environment": environment,
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(agents_data),
            "deployed_agents": len([a for a in agents_data if a["status"] == "deployed"]),
            "success_rate": 0,
            "average_pass_rate": 0,
            "agents": agents_data
        }
        
        if summary["total_agents"] > 0:
            summary["success_rate"] = (summary["deployed_agents"] / summary["total_agents"]) * 100
            pass_rates = [a["pass_rate"] for a in agents_data if a["pass_rate"] > 0]
            if pass_rates:
                summary["average_pass_rate"] = sum(pass_rates) / len(pass_rates)
        
        # Save summary
        with open(output, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Deployment summary generated: {output}")
        
    except Exception as e:
        print(f"❌ Failed to generate summary: {e}")

def collect_agent_results():
    """Collect results from training reports"""
    agents = []
    reports_dir = "training/reports"
    
    if not os.path.exists(reports_dir):
        return agents
    
    # Expected agent roles
    expected_roles = [
        "Software Engineer", "Quality Engineer", "DevOps Engineer",
        "Security Specialist", "Principal Architect", "Product Manager",
        "VP Software Products"
    ]
    
    for role in expected_roles:
        role_slug = role.replace(" ", "_").lower()
        
        # Check for evaluation report
        eval_file = f"{reports_dir}/evaluation_{role_slug}.json"
        training_file = f"{reports_dir}/training_{role_slug}.json"
        registration_file = f"{reports_dir}/registration_{role_slug}.json"
        
        agent_data = {
            "role": role,
            "status": "not_started",
            "pass_rate": 0,
            "training_time": 0,
            "deployment_time": None
        }
        
        # Check training
        if os.path.exists(training_file):
            try:
                with open(training_file, 'r') as f:
                    training_data = json.load(f)
                if training_data.get("status") == "success":
                    agent_data["status"] = "trained"
                    agent_data["training_time"] = training_data.get("training_time", 0)
                else:
                    agent_data["status"] = "training_failed"
            except:
                pass
        
        # Check evaluation
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                agent_data["pass_rate"] = eval_data.get("pass_rate", 0) * 100
                if eval_data.get("pass_rate", 0) >= 0.85:
                    agent_data["status"] = "evaluation_passed"
                else:
                    agent_data["status"] = "evaluation_failed"
            except:
                pass
        
        # Check registration
        if os.path.exists(registration_file):
            try:
                with open(registration_file, 'r') as f:
                    reg_data = json.load(f)
                if reg_data.get("status") == "success":
                    agent_data["status"] = "deployed"
                    agent_data["deployment_time"] = reg_data.get("registration_date")
            except:
                pass
        
        agents.append(agent_data)
    
    return agents

if __name__ == "__main__":
    generate_summary() 