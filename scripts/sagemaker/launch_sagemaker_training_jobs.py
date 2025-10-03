#!/usr/bin/env python3
"""
MangoMAS SageMaker Training Job Launcher
Enhanced version that integrates with Terraform infrastructure and supports all agent types.
"""

import os
import sys
import json
import time
import boto3
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role


@dataclass
class AgentTrainingConfig:
    """Configuration for individual agent training job"""
    agent_name: str
    training_file: str
    model_name: str = 'mistralai/Mistral-7B-v0.1'
    instance_type: str = 'ml.g4dn.xlarge'
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-5
    max_length: int = 512
    use_spot_instances: bool = True
    max_runtime: int = 7200  # 2 hours


class MangoMASSageMakerLauncher:
    """Enhanced SageMaker training job launcher with Terraform integration"""
    
    def __init__(self, region: str = 'us-east-1', terraform_config_path: Optional[str] = None):
        self.region = region
        self.terraform_config_path = terraform_config_path or self._find_terraform_config()
        
        # Initialize AWS clients
        self.sagemaker_session = sagemaker.Session()
        self.s3_client = boto3.client('s3', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Load Terraform configuration if available
        self.terraform_config = self._load_terraform_config()
        
        # Configure agent training jobs
        self.agent_configs = self._setup_agent_configurations()
        
        # Track training results
        self.training_results = []
        self.failed_jobs = []
        
        print(f"🎯 MangoMAS SageMaker Launcher initialized")
        print(f"📍 Region: {region}")
        print(f"🤖 Agents configured: {len(self.agent_configs)}")
    
    def _find_terraform_config(self) -> Optional[str]:
        """Find Terraform configuration file"""
        possible_paths = [
            "infrastructure/terraform/sagemaker-jobs/training_job_config.json",
            "../infrastructure/terraform/sagemaker-jobs/training_job_config.json",
            "./training_job_config.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _load_terraform_config(self) -> Optional[Dict]:
        """Load configuration from Terraform output"""
        if not self.terraform_config_path:
            print("⚠️  No Terraform config found, using default settings")
            return None
        
        try:
            with open(self.terraform_config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded Terraform config from: {self.terraform_config_path}")
            return config
        except Exception as e:
            print(f"⚠️  Could not load Terraform config: {e}")
            return None
    
    def _setup_agent_configurations(self) -> List[AgentTrainingConfig]:
        """Setup training configurations for all agents"""
        
        # Base agent configurations
        base_agents = [
            "product_manager_agent",
            "sqe_agent", 
            "architect_agent",
            "swe_agent",
            "vp_product_agent",
            "devops_agent",
            "tools_agent"
        ]
        
        # Real-world data agents
        real_world_agents = [
            "product_manager_agent_real_data",
            "sqe_agent_real_data",
            "architect_agent_real_data", 
            "swe_agent_real_data"
        ]
        
        all_agents = base_agents + real_world_agents
        
        configs = []
        for agent in all_agents:
            # Use more epochs for real-world data
            epochs = 4 if "real_data" in agent else 3
            
            config = AgentTrainingConfig(
                agent_name=agent,
                training_file=f"{agent}.jsonl",
                epochs=epochs,
                use_spot_instances=True,  # Enable cost savings
                max_runtime=7200  # 2 hours max
            )
            configs.append(config)
        
        return configs
    
    def _get_execution_role(self) -> str:
        """Get SageMaker execution role ARN"""
        # Try Terraform config first
        if self.terraform_config and 'role_arn' in self.terraform_config:
            return self.terraform_config['role_arn']
        
        # Try to get from SageMaker
        try:
            return get_execution_role()
        except Exception as e:
            print(f"⚠️  Could not get execution role: {e}")
            # Return placeholder - user must set this
            account_id = boto3.client('sts').get_caller_identity()['Account']
            return f"arn:aws:iam::{account_id}:role/mangomas-sagemaker-sagemaker-execution-role"
    
    def _get_s3_bucket(self) -> str:
        """Get S3 bucket name for training data"""
        # Try Terraform config first
        if self.terraform_config and 's3_bucket' in self.terraform_config:
            return self.terraform_config['s3_bucket']
        
        # Generate expected bucket name
        account_id = boto3.client('sts').get_caller_identity()['Account']
        return f"mangomas-sagemaker-production-{account_id}"
    
    def validate_training_data(self) -> bool:
        """Validate that all training data files exist"""
        print("🔍 Validating training data files...")
        
        missing_files = []
        total_examples = 0
        
        for config in self.agent_configs:
            file_path = f"training/{config.training_file}"
            if not os.path.exists(file_path):
                file_path = config.training_file  # Try current directory
            
            if not os.path.exists(file_path):
                missing_files.append(config.training_file)
                continue
            
            # Count examples in file
            try:
                with open(file_path, 'r') as f:
                    examples = sum(1 for line in f if line.strip())
                total_examples += examples
                print(f"  ✅ {config.training_file}: {examples} examples")
            except Exception as e:
                print(f"  ❌ {config.training_file}: Error reading file - {e}")
                missing_files.append(config.training_file)
        
        if missing_files:
            print(f"❌ Missing training files: {missing_files}")
            return False
        
        print(f"✅ All training data validated: {total_examples} total examples")
        return True
    
    def upload_training_data_to_s3(self) -> bool:
        """Upload training data to S3 bucket"""
        s3_bucket = self._get_s3_bucket()
        print(f"📤 Uploading training data to S3: {s3_bucket}")
        
        # Check if bucket exists
        try:
            self.s3_client.head_bucket(Bucket=s3_bucket)
        except:
            print(f"❌ S3 bucket not found: {s3_bucket}")
            print("💡 Run Terraform deployment first to create infrastructure")
            return False
        
        # Upload each training file
        for config in self.agent_configs:
            local_file = f"training/{config.training_file}"
            if not os.path.exists(local_file):
                local_file = config.training_file
            
            if not os.path.exists(local_file):
                print(f"⚠️  Skipping missing file: {config.training_file}")
                continue
            
            s3_key = f"datasets/{config.training_file}"
            
            try:
                self.s3_client.upload_file(local_file, s3_bucket, s3_key)
                print(f"  ✅ Uploaded: {config.training_file}")
            except Exception as e:
                print(f"  ❌ Failed to upload {config.training_file}: {e}")
                return False
        
        # Upload training script
        script_files = ["train_distilled_adapter.py", "requirements.txt"]
        for script in script_files:
            local_path = f"training/{script}" if os.path.exists(f"training/{script}") else script
            if os.path.exists(local_path):
                try:
                    self.s3_client.upload_file(local_path, s3_bucket, f"scripts/{script}")
                    print(f"  ✅ Uploaded script: {script}")
                except Exception as e:
                    print(f"  ⚠️  Could not upload {script}: {e}")
        
        print("✅ Training data upload completed")
        return True
    
    def create_huggingface_estimator(self, config: AgentTrainingConfig) -> HuggingFace:
        """Create HuggingFace estimator for training job"""
        role_arn = self._get_execution_role()
        s3_bucket = self._get_s3_bucket()
        
        # Create unique job name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{config.agent_name}-train-{timestamp}"
        
        # Hyperparameters
        hyperparameters = {
            'model_name_or_path': config.model_name,
            'train_file': config.training_file,
            'num_train_epochs': str(config.epochs),
            'per_device_train_batch_size': str(config.batch_size),
            'learning_rate': str(config.learning_rate),
            'max_length': str(config.max_length),
            'output_dir': '/opt/ml/model',
            'logging_dir': '/opt/ml/output/logs',
            'logging_steps': '100',
            'save_steps': '500',
            'save_total_limit': '2',
            'evaluation_strategy': 'steps',
            'eval_steps': '500',
            'load_best_model_at_end': 'True',
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': 'False',
            'warmup_steps': '100',
            'weight_decay': '0.01',
            'fp16': 'True',
            'dataloader_num_workers': '4',
            'remove_unused_columns': 'False',
            'push_to_hub': 'False'
        }
        
        # Create estimator
        estimator = HuggingFace(
            entry_point='train_distilled_adapter.py',
            source_dir=f's3://{s3_bucket}/scripts/',
            instance_type=config.instance_type,
            instance_count=1,
            role=role_arn,
            transformers_version='4.26.0',
            pytorch_version='1.13.1',
            py_version='py39',
            hyperparameters=hyperparameters,
            base_job_name=config.agent_name,
            use_spot_instances=config.use_spot_instances,
            max_run=config.max_runtime,
            output_path=f"s3://{s3_bucket}/models/{config.agent_name}/",
            code_location=f"s3://{s3_bucket}/code/{config.agent_name}/",
            tags=[
                {'Key': 'Project', 'Value': 'MangoMAS'},
                {'Key': 'AgentType', 'Value': config.agent_name},
                {'Key': 'Environment', 'Value': 'production'},
                {'Key': 'TrainingSystem', 'Value': 'agent-distillation'},
                {'Key': 'ModelType', 'Value': 'mistral-7b-distilled'},
                {'Key': 'CostOptimized', 'Value': str(config.use_spot_instances)}
            ]
        )
        
        return estimator
    
    async def launch_training_job(self, config: AgentTrainingConfig) -> Dict[str, Any]:
        """Launch individual training job"""
        print(f"\n🚀 Starting training job: {config.agent_name}")
        print(f"📁 Training file: {config.training_file}")
        print(f"⚙️  Instance: {config.instance_type}")
        print(f"💰 Spot instances: {config.use_spot_instances}")
        
        start_time = time.time()
        
        try:
            # Create estimator
            estimator = self.create_huggingface_estimator(config)
            
            # Prepare training data input
            s3_bucket = self._get_s3_bucket()
            training_input = f"s3://{s3_bucket}/datasets/{config.training_file}"
            
            # Start training
            print(f"⏳ Launching SageMaker training job...")
            estimator.fit({'training': training_input}, wait=False)
            
            training_time = time.time() - start_time
            job_name = estimator.latest_training_job.name
            
            result = {
                'agent_name': config.agent_name,
                'job_name': job_name,
                'status': 'started',
                'training_time': training_time,
                'training_input': training_input,
                'model_output': estimator.output_path,
                'instance_type': config.instance_type,
                'use_spot': config.use_spot_instances,
                'hyperparameters': estimator.hyperparameters(),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Training job started: {job_name}")
            print(f"⏱️  Setup time: {training_time:.2f} seconds")
            print(f"📦 Model output: {estimator.output_path}")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            error_msg = f"Failed to start training job for {config.agent_name}: {str(e)}"
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
        
        # Validate training data
        if not self.validate_training_data():
            print("❌ Training data validation failed")
            return []
        
        # Upload training data to S3
        if not self.upload_training_data_to_s3():
            print("❌ Failed to upload training data")
            return []
        
        print(f"\n🚀 Launching {len(self.agent_configs)} training jobs...")
        print(f"🔄 Parallel execution: {parallel}")
        print(f"📊 Max concurrent jobs: {max_concurrent}")
        
        if parallel:
            # Launch jobs with concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def launch_with_semaphore(config):
                async with semaphore:
                    return await self.launch_training_job(config)
            
            tasks = [
                launch_with_semaphore(config) 
                for config in self.agent_configs
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        else:
            # Launch jobs sequentially
            results = []
            for config in self.agent_configs:
                result = await self.launch_training_job(config)
                results.append(result)
                # Small delay between jobs
                await asyncio.sleep(5)
        
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
    
    def monitor_training_jobs(self, job_names: List[str] = None) -> Dict[str, str]:
        """Monitor status of training jobs"""
        if not job_names:
            job_names = [result['job_name'] for result in self.training_results]
        
        print("\n📊 Monitoring training jobs...")
        statuses = {}
        
        for job_name in job_names:
            try:
                response = self.sagemaker_client.describe_training_job(
                    TrainingJobName=job_name
                )
                status = response['TrainingJobStatus']
                statuses[job_name] = status
                
                # Color code status
                if status == 'Completed':
                    status_display = f"✅ {status}"
                elif status == 'Failed':
                    status_display = f"❌ {status}"
                elif status == 'InProgress':
                    status_display = f"⏳ {status}"
                else:
                    status_display = f"🔄 {status}"
                
                print(f"  {job_name}: {status_display}")
                
            except Exception as e:
                print(f"  {job_name}: ❌ Error checking status - {e}")
                statuses[job_name] = 'Error'
        
        return statuses
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive training summary"""
        total_jobs = len(self.agent_configs)
        started_jobs = len(self.training_results)
        failed_jobs = len(self.failed_jobs)
        success_rate = (started_jobs / total_jobs) * 100 if total_jobs > 0 else 0
        
        report = f"""
🎯 MangoMAS SageMaker Training Summary
{'=' * 50}

📊 Launch Statistics:
- Total Jobs: {total_jobs}
- Successfully Started: {started_jobs}
- Failed to Start: {failed_jobs}
- Launch Success Rate: {success_rate:.1f}%

🚀 Started Training Jobs:
"""
        
        for result in self.training_results:
            report += f"""
- {result['agent_name']}:
  - Job Name: {result['job_name']}
  - Instance: {result['instance_type']}
  - Spot Training: {result['use_spot']}
  - Model Output: {result['model_output']}
"""
        
        if self.failed_jobs:
            report += f"""
❌ Failed Jobs:
"""
            for job in self.failed_jobs:
                report += f"- {job.get('agent_name', 'Unknown')}: {job.get('error', 'Unknown error')}\n"
        
        # Cost estimation
        spot_jobs = sum(1 for r in self.training_results if r.get('use_spot', False))
        regular_jobs = started_jobs - spot_jobs
        
        estimated_cost_spot = spot_jobs * 0.50  # ~$0.50 per job with spot
        estimated_cost_regular = regular_jobs * 5.00  # ~$5.00 per job regular
        total_estimated_cost = estimated_cost_spot + estimated_cost_regular
        
        report += f"""
💰 Cost Estimation:
- Spot Instance Jobs: {spot_jobs} (~${estimated_cost_spot:.2f})
- Regular Instance Jobs: {regular_jobs} (~${estimated_cost_regular:.2f})
- Total Estimated Cost: ~${total_estimated_cost:.2f}

📋 Next Steps:
1. Monitor training progress in SageMaker console
2. Check CloudWatch logs for detailed training info
3. Download trained models from S3 when complete
4. Deploy models to inference endpoints
5. Integrate with MangoMAS application

🕐 Report generated: {datetime.now().isoformat()}
"""
        
        return report
    
    def save_results(self, filename: str = "sagemaker_training_results.json"):
        """Save training results to file"""
        results = {
            'summary': {
                'total_jobs': len(self.agent_configs),
                'started_jobs': len(self.training_results),
                'failed_jobs': len(self.failed_jobs),
                'success_rate': len(self.training_results) / len(self.agent_configs) if self.agent_configs else 0,
                'timestamp': datetime.now().isoformat()
            },
            'started_jobs': self.training_results,
            'failed_jobs': self.failed_jobs,
            'terraform_config': self.terraform_config
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")


async def main():
    """Main function to launch all training jobs"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch MangoMAS SageMaker training jobs')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--parallel', action='store_true', default=True, help='Launch jobs in parallel')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Max concurrent jobs')
    parser.add_argument('--terraform-config', help='Path to Terraform config file')
    parser.add_argument('--monitor', action='store_true', help='Monitor existing jobs instead of launching new ones')
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = MangoMASSageMakerLauncher(
        region=args.region,
        terraform_config_path=args.terraform_config
    )
    
    if args.monitor:
        # Monitor existing jobs
        launcher.monitor_training_jobs()
    else:
        # Launch all jobs
        results = await launcher.launch_all_jobs(
            parallel=args.parallel,
            max_concurrent=args.max_concurrent
        )
        
        # Generate and print summary
        summary = launcher.generate_summary_report()
        print(summary)
        
        # Save results
        launcher.save_results()
        
        # Return results
        return results


if __name__ == "__main__":
    # Run the launcher
    try:
        results = asyncio.run(main())
        
        if results:
            print("🎉 Training jobs launched successfully!")
            print(f"📊 {len(results)} jobs started")
            print("💡 Use --monitor flag to check job status")
        else:
            print("⚠️  No training jobs were started")
            
    except KeyboardInterrupt:
        print("\n⚠️  Launch interrupted by user")
    except Exception as e:
        print(f"❌ Error during launch: {e}")
        sys.exit(1) 