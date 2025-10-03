#!/usr/bin/env python3
"""
MangoMAS Multi-Agent SageMaker Training Launcher
Launches separate SageMaker training jobs for each agent type using real-world training data.
"""

import os
import sys
import time
import json
import boto3
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role

@dataclass
class TrainingJobConfig:
    """Configuration for individual training job"""
    agent_name: str
    training_file: str
    model_name: str = 'mistralai/Mistral-7B-v0.1'
    instance_type: str = 'ml.g4dn.xlarge'
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-5
    max_length: int = 512

class MangoMASSageMakerLauncher:
    """Launcher for multiple SageMaker training jobs"""
    
    def __init__(self, region: str = 'us-east-1', role_arn: Optional[str] = None):
        self.region = region
        self.role_arn = role_arn or self._get_default_role()
        self.sagemaker_session = sagemaker.Session()
        self.s3_client = boto3.client('s3', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Training data files (using real-world data we created)
        self.agent_configs = [
            TrainingJobConfig(
                agent_name="product_manager",
                training_file="product_manager_agent_real_data.jsonl",
                epochs=3,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="sqe",
                training_file="sqe_agent_real_data.jsonl", 
                epochs=3,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="architect",
                training_file="architect_agent_real_data.jsonl",
                epochs=3,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="swe",
                training_file="swe_agent_real_data.jsonl",
                epochs=3,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="product_manager_basic",
                training_file="product_manager_agent.jsonl",
                epochs=2,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="sqe_basic",
                training_file="sqe_agent.jsonl",
                epochs=2,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="architect_basic",
                training_file="architect_agent.jsonl",
                epochs=2,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="swe_basic",
                training_file="swe_agent.jsonl",
                epochs=2,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="vp_product",
                training_file="vp_product_agent.jsonl",
                epochs=2,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="devops",
                training_file="devops_agent.jsonl",
                epochs=2,
                batch_size=2
            ),
            TrainingJobConfig(
                agent_name="tools",
                training_file="tools_agent.jsonl",
                epochs=2,
                batch_size=2
            )
        ]
        
        self.training_results = []
        self.failed_jobs = []
        
    def _get_default_role(self) -> str:
        """Get default SageMaker execution role"""
        try:
            return get_execution_role()
        except Exception as e:
            print(f"⚠️  Warning: Could not get default role: {e}")
            print("Please set role_arn parameter or configure AWS credentials")
            return "arn:aws:iam::<your-account-id>:role/SageMakerExecutionRole"
    
    def validate_training_files(self) -> bool:
        """Validate that all training files exist"""
        print("🔍 Validating training files...")
        missing_files = []
        
        for config in self.agent_configs:
            file_path = Path(f"training/{config.training_file}")
            if not file_path.exists():
                missing_files.append(config.training_file)
                print(f"❌ Missing: {config.training_file}")
            else:
                # Count lines in file
                with open(file_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                print(f"✅ Found: {config.training_file} ({line_count} training examples)")
        
        if missing_files:
            print(f"\n❌ Missing training files: {missing_files}")
            return False
        
        print(f"✅ All {len(self.agent_configs)} training files validated")
        return True
    
    def upload_training_data_to_s3(self, bucket_name: str = "mangomas-training-data") -> Dict[str, str]:
        """Upload training files to S3 and return S3 URIs"""
        print(f"📤 Uploading training data to S3 bucket: {bucket_name}")
        
        # Create bucket if it doesn't exist
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
        except:
            print(f"Creating S3 bucket: {bucket_name}")
            self.s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.region}
            )
        
        s3_uris = {}
        
        for config in self.agent_configs:
            local_path = f"training/{config.training_file}"
            s3_key = f"training-data/{config.training_file}"
            
            try:
                self.s3_client.upload_file(local_path, bucket_name, s3_key)
                s3_uri = f"s3://{bucket_name}/{s3_key}"
                s3_uris[config.agent_name] = s3_uri
                print(f"✅ Uploaded: {config.training_file} -> {s3_uri}")
            except Exception as e:
                print(f"❌ Failed to upload {config.training_file}: {e}")
                return {}
        
        return s3_uris
    
    def create_huggingface_estimator(self, config: TrainingJobConfig, s3_uri: str) -> HuggingFace:
        """Create HuggingFace estimator for training job"""
        
        # Create unique job name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"mangomas-{config.agent_name}-{timestamp}"
        
        # Hyperparameters
        hyperparameters = {
            'model_name_or_path': config.model_name,
            'train_file': config.training_file,
            'num_train_epochs': config.epochs,
            'per_device_train_batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'max_length': config.max_length,
            'output_dir': '/opt/ml/model',
            'logging_steps': 100,
            'save_steps': 500,
            'save_total_limit': 2,
            'evaluation_strategy': 'steps',
            'eval_steps': 500,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'warmup_steps': 100,
            'weight_decay': 0.01,
            'fp16': True,
            'dataloader_num_workers': 4,
            'remove_unused_columns': False,
            'push_to_hub': False
        }
        
        # Create estimator
        estimator = HuggingFace(
            entry_point='train_distilled_adapter.py',
            source_dir='training',
            instance_type=config.instance_type,
            instance_count=1,
            role=self.role_arn,
            transformers_version='4.26.0',
            pytorch_version='1.13.1',
            py_version='py39',
            hyperparameters=hyperparameters,
            base_job_name=job_name,
            region=self.region,
            sagemaker_session=self.sagemaker_session,
            output_path=f"s3://mangomas-training-data/models/{config.agent_name}",
            code_location=f"s3://mangomas-training-data/code/{config.agent_name}",
            max_run=24*60*60,  # 24 hours
            keep_alive_period_in_seconds=1800,  # 30 minutes
            tags=[
                {'Key': 'Project', 'Value': 'MangoMAS'},
                {'Key': 'AgentType', 'Value': config.agent_name},
                {'Key': 'TrainingData', 'Value': 'real-world'},
                {'Key': 'ModelType', 'Value': 'distilled-adapter'}
            ]
        )
        
        return estimator
    
    async def launch_training_job(self, config: TrainingJobConfig, s3_uri: str) -> Dict:
        """Launch individual training job"""
        print(f"\n🚀 Starting training job for {config.agent_name}")
        print(f"📁 Training file: {config.training_file}")
        print(f"🔗 S3 URI: {s3_uri}")
        
        start_time = time.time()
        
        try:
            # Create estimator
            estimator = self.create_huggingface_estimator(config, s3_uri)
            
            # Start training
            print(f"⏳ Launching SageMaker training job...")
            estimator.fit({'training': s3_uri})
            
            # Get training results
            training_time = time.time() - start_time
            job_name = estimator.latest_training_job.name
            
            result = {
                'agent_name': config.agent_name,
                'job_name': job_name,
                'status': 'completed',
                'training_time': training_time,
                's3_uri': s3_uri,
                'model_artifact': estimator.model_data,
                'hyperparameters': estimator.hyperparameters(),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Completed training job for {config.agent_name}")
            print(f"⏱️  Training time: {training_time:.2f} seconds")
            print(f"📦 Model artifact: {estimator.model_data}")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            error_msg = f"Training job failed for {config.agent_name}: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                'agent_name': config.agent_name,
                'status': 'failed',
                'error': str(e),
                'training_time': training_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def launch_all_jobs(self, parallel: bool = True, max_concurrent: int = 3) -> List[Dict]:
        """Launch all training jobs"""
        print("🎯 MangoMAS Multi-Agent SageMaker Training Launcher")
        print("=" * 60)
        
        # Validate files
        if not self.validate_training_files():
            print("❌ Training file validation failed")
            return []
        
        # Upload to S3
        s3_uris = self.upload_training_data_to_s3()
        if not s3_uris:
            print("❌ Failed to upload training data to S3")
            return []
        
        print(f"\n🚀 Launching {len(self.agent_configs)} training jobs...")
        print(f"🔄 Parallel execution: {parallel}")
        print(f"📊 Max concurrent jobs: {max_concurrent}")
        
        if parallel:
            # Launch jobs with concurrency limit
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def launch_with_semaphore(config):
                async with semaphore:
                    return await self.launch_training_job(config, s3_uris[config.agent_name])
            
            tasks = [
                launch_with_semaphore(config) 
                for config in self.agent_configs
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        else:
            # Launch jobs sequentially
            results = []
            for config in self.agent_configs:
                result = await self.launch_training_job(config, s3_uris[config.agent_name])
                results.append(result)
        
        # Process results
        self.training_results = []
        self.failed_jobs = []
        
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Exception in training job: {result}")
                self.failed_jobs.append({'error': str(result)})
            elif result['status'] == 'failed':
                self.failed_jobs.append(result)
            else:
                self.training_results.append(result)
        
        return self.training_results
    
    def generate_summary_report(self) -> str:
        """Generate training summary report"""
        total_jobs = len(self.agent_configs)
        successful_jobs = len(self.training_results)
        failed_jobs = len(self.failed_jobs)
        
        report = f"""
🎯 MangoMAS Training Summary Report
{'=' * 50}

📊 Overall Statistics:
- Total Jobs Launched: {total_jobs}
- Successful Jobs: {successful_jobs}
- Failed Jobs: {failed_jobs}
- Success Rate: {(successful_jobs/total_jobs)*100:.1f}%

✅ Successful Training Jobs:
"""
        
        for result in self.training_results:
            report += f"""
- {result['agent_name']}:
  - Job Name: {result['job_name']}
  - Training Time: {result['training_time']:.2f}s
  - Model Artifact: {result['model_artifact']}
  - Status: {result['status']}
"""
        
        if self.failed_jobs:
            report += f"""
❌ Failed Training Jobs:
"""
            for job in self.failed_jobs:
                report += f"- {job.get('agent_name', 'Unknown')}: {job.get('error', 'Unknown error')}\n"
        
        report += f"""
📈 Next Steps:
1. Monitor training jobs in SageMaker console
2. Download trained models from S3
3. Deploy models to inference endpoints
4. Test agent performance with real scenarios

🕐 Report generated: {datetime.now().isoformat()}
"""
        
        return report
    
    def save_results(self, filename: str = "training_results.json"):
        """Save training results to file"""
        results = {
            'summary': {
                'total_jobs': len(self.agent_configs),
                'successful_jobs': len(self.training_results),
                'failed_jobs': len(self.failed_jobs),
                'success_rate': len(self.training_results) / len(self.agent_configs),
                'timestamp': datetime.now().isoformat()
            },
            'successful_jobs': self.training_results,
            'failed_jobs': self.failed_jobs
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")

async def main():
    """Main function to launch all training jobs"""
    
    # Configuration
    region = os.getenv('AWS_REGION', 'us-east-1')
    role_arn = os.getenv('SAGEMAKER_ROLE_ARN')  # Optional: set via environment variable
    
    # Create launcher
    launcher = MangoMASSageMakerLauncher(region=region, role_arn=role_arn)
    
    # Launch all jobs
    results = await launcher.launch_all_jobs(parallel=True, max_concurrent=3)
    
    # Generate and print summary
    summary = launcher.generate_summary_report()
    print(summary)
    
    # Save results
    launcher.save_results()
    
    return results

if __name__ == "__main__":
    # Run the launcher
    results = asyncio.run(main())
    
    # Exit with appropriate code
    if results and len(results) == len(launcher.agent_configs):
        print("🎉 All training jobs completed successfully!")
        sys.exit(0)
    else:
        print("⚠️  Some training jobs failed. Check the summary above.")
        sys.exit(1) 