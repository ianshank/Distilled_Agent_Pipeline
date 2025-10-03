#!/usr/bin/env python3
"""
Simple MangoMAS SageMaker Training Job Launcher
Based on user template but enhanced with production features.
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from datetime import datetime

def get_role_and_bucket():
    """Get IAM role and S3 bucket from AWS account"""
    # Get AWS account ID
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    
    # Construct role ARN and bucket name
    role = f"arn:aws:iam::{account_id}:role/mangomas-sagemaker-sagemaker-execution-role"
    bucket = f"mangomas-sagemaker-production-{account_id}"
    
    return role, bucket

def launch_sagemaker_training_jobs():
    """Launch SageMaker training jobs for all MangoMAS agents"""
    
    # Configuration
    region = "us-east-1"
    role, s3_bucket = get_role_and_bucket()
    
    print(f"🎯 MangoMAS SageMaker Training Launcher")
    print(f"📍 Region: {region}")
    print(f"🔐 Role: {role}")
    print(f"📦 S3 Bucket: {s3_bucket}")
    print("=" * 60)
    
    # Agent training files (including real-world data)
    agent_files = [
        "product_manager_agent.jsonl",
        "sqe_agent.jsonl",
        "architect_agent.jsonl", 
        "swe_agent.jsonl",
        "vp_product_agent.jsonl",
        "devops_agent.jsonl",
        "tools_agent.jsonl",
        "product_manager_agent_real_data.jsonl",
        "sqe_agent_real_data.jsonl",
        "architect_agent_real_data.jsonl",
        "swe_agent_real_data.jsonl"
    ]
    
    training_jobs = []
    
    for agent_file in agent_files:
        agent_name = agent_file.replace(".jsonl", "").replace("_", "-")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        print(f"\n🚀 Starting training job for {agent_name}")
        print(f"📁 Training file: {agent_file}")
        
        try:
            # Create HuggingFace estimator
            huggingface_estimator = HuggingFace(
                entry_point='train_distilled_adapter.py',
                source_dir=f's3://{s3_bucket}/scripts/',
                instance_type='ml.g4dn.xlarge',
                instance_count=1,
                role=role,
                transformers_version='4.26.0',
                pytorch_version='1.13.1',
                py_version='py39',
                use_spot_instances=True,  # 90% cost savings
                max_run=7200,  # 2 hours max
                hyperparameters={
                    'model_name_or_path': 'mistralai/Mistral-7B-v0.1',
                    'train_file': agent_file,
                    'num_train_epochs': '4' if 'real_data' in agent_file else '3',
                    'per_device_train_batch_size': '2',
                    'learning_rate': '2e-5',
                    'output_dir': '/opt/ml/model',
                    'logging_dir': '/opt/ml/output/logs',
                    'logging_steps': '100',
                    'save_steps': '500',
                    'fp16': 'True',
                    'push_to_hub': 'False'
                },
                base_job_name=f"{agent_name}-train",
                output_path=f"s3://{s3_bucket}/models/{agent_name}/",
                tags=[
                    {'Key': 'Project', 'Value': 'MangoMAS'},
                    {'Key': 'AgentType', 'Value': agent_name},
                    {'Key': 'Environment', 'Value': 'production'},
                    {'Key': 'TrainingSystem', 'Value': 'agent-distillation'},
                    {'Key': 'CostOptimized', 'Value': 'True'}
                ]
            )
            
            # Start training (async)
            training_input = f"s3://{s3_bucket}/datasets/{agent_file}"
            huggingface_estimator.fit({'training': training_input}, wait=False)
            
            job_name = huggingface_estimator.latest_training_job.name
            training_jobs.append({
                'agent_name': agent_name,
                'job_name': job_name,
                'training_file': agent_file,
                'model_output': huggingface_estimator.output_path
            })
            
            print(f"✅ Training job started: {job_name}")
            print(f"📦 Model output: {huggingface_estimator.output_path}")
            
        except Exception as e:
            print(f"❌ Failed to start training job for {agent_name}: {e}")
    
    # Summary
    print(f"\n🎉 Training Job Launch Summary")
    print("=" * 60)
    print(f"📊 Total jobs launched: {len(training_jobs)}")
    print(f"💰 Cost optimization: Spot instances enabled (90% savings)")
    print(f"⏱️  Max runtime: 2 hours per job")
    
    if training_jobs:
        print(f"\n🚀 Successfully launched jobs:")
        for job in training_jobs:
            print(f"  ✅ {job['agent_name']}: {job['job_name']}")
        
        print(f"\n📋 Next steps:")
        print(f"1. Monitor jobs in SageMaker console")
        print(f"2. Check CloudWatch logs: /aws/sagemaker/TrainingJobs")
        print(f"3. Download models from S3: s3://{s3_bucket}/models/")
        print(f"4. Deploy models to inference endpoints")
    
    return training_jobs

if __name__ == "__main__":
    print("🎯 MangoMAS Simple SageMaker Launcher")
    print("🚀 Launching all agent training jobs...")
    
    try:
        jobs = launch_sagemaker_training_jobs()
        print(f"\n🎉 All done! {len(jobs)} training jobs launched.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have:")
        print("  - AWS credentials configured")
        print("  - SageMaker execution role created")
        print("  - S3 bucket with training data")
        print("  - Run Terraform deployment first") 