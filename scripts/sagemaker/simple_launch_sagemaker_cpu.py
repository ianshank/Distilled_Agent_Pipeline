#!/usr/bin/env python3
"""
Simple MangoMAS SageMaker Training Job Launcher - CPU Version
Uses CPU instances with higher default quotas for initial testing.
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
    """Launch SageMaker training jobs using CPU instances with higher quotas"""
    
    # Configuration
    region = "us-east-1"
    role, s3_bucket = get_role_and_bucket()
    
    print(f"🎯 MangoMAS SageMaker Training Launcher (CPU Version)")
    print(f"📍 Region: {region}")
    print(f"🔐 Role: {role}")
    print(f"📦 S3 Bucket: {s3_bucket}")
    print(f"⚡ Instance: ml.m5.large (CPU - Higher default quota)")
    print("=" * 60)
    
    # Start with a smaller subset to test quotas
    agent_files = [
        "product_manager_agent.jsonl",
        "sqe_agent.jsonl",
        "architect_agent.jsonl"
    ]
    
    training_jobs = []
    
    for agent_file in agent_files:
        agent_name = agent_file.replace(".jsonl", "").replace("_", "-")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        print(f"\n🚀 Starting training job for {agent_name}")
        print(f"📁 Training file: {agent_file}")
        
        try:
            # Create HuggingFace estimator with CPU instance
            huggingface_estimator = HuggingFace(
                entry_point='train_distilled_adapter.py',
                source_dir=f's3://{s3_bucket}/scripts/',
                instance_type='ml.m5.large',  # CPU instance with higher quota
                instance_count=1,
                role=role,
                transformers_version='4.26.0',
                pytorch_version='1.13.1',
                py_version='py39',
                use_spot_instances=False,  # Disable spot for initial test
                max_run=3600,  # 1 hour max for CPU training
                hyperparameters={
                    'model_name_or_path': 'distilbert-base-uncased',  # Smaller model for CPU
                    'train_file': agent_file,
                    'num_train_epochs': '1',  # Fewer epochs for CPU
                    'per_device_train_batch_size': '8',  # Larger batch for CPU
                    'learning_rate': '5e-5',
                    'output_dir': '/opt/ml/model',
                    'logging_dir': '/opt/ml/output/logs',
                    'logging_steps': '50',
                    'save_steps': '100',
                    'max_seq_length': '128',  # Shorter sequences for CPU
                    'fp16': 'False',  # No mixed precision on CPU
                    'push_to_hub': 'False'
                },
                base_job_name=f"{agent_name}-cpu-train",
                output_path=f"s3://{s3_bucket}/models/{agent_name}/",
                tags=[
                    {'Key': 'Project', 'Value': 'MangoMAS'},
                    {'Key': 'AgentType', 'Value': agent_name},
                    {'Key': 'Environment', 'Value': 'production'},
                    {'Key': 'TrainingSystem', 'Value': 'agent-distillation'},
                    {'Key': 'InstanceType', 'Value': 'CPU'},
                    {'Key': 'TestRun', 'Value': 'True'}
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
                'model_output': huggingface_estimator.output_path,
                'instance_type': 'ml.m5.large'
            })
            
            print(f"✅ Training job started: {job_name}")
            print(f"📦 Model output: {huggingface_estimator.output_path}")
            print(f"⚡ Instance: ml.m5.large (CPU)")
            
        except Exception as e:
            print(f"❌ Failed to start training job for {agent_name}: {e}")
    
    # Summary
    print(f"\n🎉 CPU Training Job Launch Summary")
    print("=" * 60)
    print(f"📊 Total jobs launched: {len(training_jobs)}")
    print(f"⚡ Instance type: ml.m5.large (CPU)")
    print(f"🧠 Model: DistilBERT (CPU-optimized)")
    print(f"⏱️  Max runtime: 1 hour per job")
    print(f"💰 Cost: ~$0.10 per job (no spot pricing needed)")
    
    if training_jobs:
        print(f"\n🚀 Successfully launched jobs:")
        for job in training_jobs:
            print(f"  ✅ {job['agent_name']}: {job['job_name']}")
        
        print(f"\n📋 Next steps:")
        print(f"1. Monitor jobs in SageMaker console")
        print(f"2. Check CloudWatch logs: /aws/sagemaker/TrainingJobs")
        print(f"3. Download models from S3: s3://{s3_bucket}/models/")
        print(f"4. Request GPU instance quota increase for full training")
        print(f"\n💡 To request quota increase:")
        print(f"  - Go to AWS Service Quotas console")
        print(f"  - Search for 'SageMaker'")
        print(f"  - Request increase for 'ml.g4dn.xlarge for spot training job usage'")
        print(f"  - Recommend requesting 5-10 instances for parallel training")
    
    return training_jobs

if __name__ == "__main__":
    print("🎯 MangoMAS CPU SageMaker Launcher")
    print("🚀 Testing with CPU instances (higher default quotas)...")
    
    try:
        jobs = launch_sagemaker_training_jobs()
        if jobs:
            print(f"\n🎉 SUCCESS! {len(jobs)} CPU training jobs launched.")
            print(f"💡 This proves the infrastructure works - just need GPU quota increase!")
        else:
            print(f"\n⚠️  No training jobs launched.")
            print(f"💡 May need to request quota increases for CPU instances too.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Troubleshooting:")
        print("  - Check AWS credentials and permissions")
        print("  - Verify Terraform deployment completed successfully")
        print("  - Ensure training data files exist") 