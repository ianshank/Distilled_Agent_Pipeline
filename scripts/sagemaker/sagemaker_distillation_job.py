#!/usr/bin/env python3
"""
MangoMAS SageMaker Agent Distillation Training Job
Configures and launches SageMaker training jobs for agent distillation
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class AgentDistillationJob:
    """SageMaker training job manager for agent distillation"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.sagemaker_session = sagemaker.Session()
        self.role = get_execution_role()
        self.s3_bucket = self.sagemaker_session.default_bucket()
        
    def create_training_job(self, 
                           job_name: str,
                           teacher_model: str,
                           student_model: str,
                           train_data_s3: str,
                           eval_data_s3: Optional[str] = None,
                           instance_type: str = 'ml.g5.2xlarge',
                           instance_count: int = 1,
                           hyperparameters: Optional[Dict[str, Any]] = None,
                           use_spot_instances: bool = True,
                           max_wait_time: int = 3600,
                           **kwargs) -> str:
        """
        Create and launch a SageMaker training job for agent distillation
        
        Args:
            job_name: Name of the training job
            teacher_model: HuggingFace model name for teacher
            student_model: HuggingFace model name for student
            train_data_s3: S3 URI for training data
            eval_data_s3: S3 URI for evaluation data (optional)
            instance_type: SageMaker instance type
            instance_count: Number of instances
            hyperparameters: Training hyperparameters
            use_spot_instances: Whether to use spot instances
            max_wait_time: Maximum wait time in seconds
            **kwargs: Additional arguments
            
        Returns:
            Training job name
        """
        
        # Default hyperparameters
        default_hyperparameters = {
            'teacher_model_name': teacher_model,
            'student_model_name': student_model,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_accumulation_steps': 4,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'distillation_alpha': 0.5,
            'temperature': 2.0,
            'max_length': 512,
            'logging_steps': 100,
            'save_steps': 500,
            'save_total_limit': 2,
            'eval_steps': 500,
            'use_lora': True,
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'lora_target_modules': 'q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
            'use_fp16': True,
            'use_device_map': True,
            'use_wandb': False
        }
        
        # Update with provided hyperparameters
        if hyperparameters:
            default_hyperparameters.update(hyperparameters)
        
        # Add evaluation file if provided
        if eval_data_s3:
            default_hyperparameters['eval_file'] = '/opt/ml/input/data/eval/train.jsonl'
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point='train_distilled_adapter.py',
            source_dir='training',
            role=self.role,
            instance_count=instance_count,
            instance_type=instance_type,
            framework_version='2.0.1',
            py_version='py310',
            hyperparameters=default_hyperparameters,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait_time,
            max_run=max_wait_time,
            output_path=f's3://{self.s3_bucket}/mangomas-distillation-output',
            code_location=f's3://{self.s3_bucket}/mangomas-distillation-code',
            base_job_name=job_name,
            **kwargs
        )
        
        # Prepare input data configuration
        inputs = {
            'train': sagemaker.inputs.TrainingInput(
                s3_data=train_data_s3,
                content_type='application/json'
            )
        }
        
        if eval_data_s3:
            inputs['eval'] = sagemaker.inputs.TrainingInput(
                s3_data=eval_data_s3,
                content_type='application/json'
            )
        
        # Launch training job
        estimator.fit(inputs, job_name=job_name)
        
        return job_name
    
    def create_distillation_pipeline(self,
                                   pipeline_name: str,
                                   teacher_model: str,
                                   student_model: str,
                                   train_data_s3: str,
                                   eval_data_s3: Optional[str] = None,
                                   **kwargs) -> str:
        """
        Create a complete distillation pipeline with multiple stages
        
        Args:
            pipeline_name: Name of the pipeline
            teacher_model: Teacher model name
            student_model: Student model name
            train_data_s3: Training data S3 URI
            eval_data_s3: Evaluation data S3 URI
            **kwargs: Additional arguments
            
        Returns:
            Pipeline name
        """
        
        # Stage 1: Initial distillation
        stage1_job = f"{pipeline_name}-stage1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        stage1_hyperparameters = {
            'distillation_alpha': 0.7,
            'temperature': 3.0,
            'num_train_epochs': 2,
            'learning_rate': 1e-4
        }
        
        self.create_training_job(
            job_name=stage1_job,
            teacher_model=teacher_model,
            student_model=student_model,
            train_data_s3=train_data_s3,
            eval_data_s3=eval_data_s3,
            hyperparameters=stage1_hyperparameters,
            **kwargs
        )
        
        # Stage 2: Fine-tuning
        stage2_job = f"{pipeline_name}-stage2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        stage2_hyperparameters = {
            'distillation_alpha': 0.3,
            'temperature': 1.5,
            'num_train_epochs': 1,
            'learning_rate': 5e-5
        }
        
        # Use output from stage 1 as input for stage 2
        stage1_output = f's3://{self.s3_bucket}/mangomas-distillation-output/{stage1_job}/output/model.tar.gz'
        
        self.create_training_job(
            job_name=stage2_job,
            teacher_model=teacher_model,
            student_model=stage1_output,  # Use stage 1 output
            train_data_s3=train_data_s3,
            eval_data_s3=eval_data_s3,
            hyperparameters=stage2_hyperparameters,
            **kwargs
        )
        
        return pipeline_name
    
    def monitor_training_job(self, job_name: str):
        """Monitor training job progress"""
        client = boto3.client('sagemaker', region_name=self.region)
        
        try:
            response = client.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            print(f"Training Job: {job_name}")
            print(f"Status: {status}")
            
            if 'FinalMetricDataList' in response:
                print("\nFinal Metrics:")
                for metric in response['FinalMetricDataList']:
                    print(f"  {metric['MetricName']}: {metric['Value']}")
            
            if 'OutputDataConfig' in response:
                output_location = response['OutputDataConfig']['S3OutputPath']
                print(f"\nOutput Location: {output_location}")
                
        except Exception as e:
            print(f"Error monitoring job: {e}")
    
    def list_training_jobs(self, name_contains: str = 'mangomas'):
        """List training jobs with optional filtering"""
        client = boto3.client('sagemaker', region_name=self.region)
        
        try:
            response = client.list_training_jobs(
                NameContains=name_contains,
                MaxResults=20
            )
            
            print(f"Training Jobs containing '{name_contains}':")
            for job in response['TrainingJobSummaries']:
                print(f"  {job['TrainingJobName']} - {job['TrainingJobStatus']} - {job['CreationTime']}")
                
        except Exception as e:
            print(f"Error listing jobs: {e}")
    
    def download_model(self, job_name: str, local_path: str):
        """Download trained model from S3"""
        import tarfile
        
        # Get model location
        client = boto3.client('sagemaker', region_name=self.region)
        response = client.describe_training_job(TrainingJobName=job_name)
        model_data = response['ModelArtifacts']['S3ModelArtifacts']
        
        # Download and extract
        s3_client = boto3.client('s3')
        
        # Parse S3 URI
        bucket = model_data.split('/')[2]
        key = '/'.join(model_data.split('/')[3:])
        
        # Download
        local_tar = os.path.join(local_path, 'model.tar.gz')
        s3_client.download_file(bucket, key, local_tar)
        
        # Extract
        with tarfile.open(local_tar, 'r:gz') as tar:
            tar.extractall(local_path)
        
        print(f"Model downloaded to: {local_path}")
    
    def create_inference_endpoint(self, 
                                model_name: str,
                                endpoint_name: str,
                                instance_type: str = 'ml.m5.large',
                                instance_count: int = 1):
        """Create SageMaker inference endpoint for distilled model"""
        
        # Create model
        model = sagemaker.pytorch.PyTorchModel(
            model_data=f's3://{self.s3_bucket}/mangomas-distillation-output/{model_name}/output/model.tar.gz',
            role=self.role,
            entry_point='inference.py',
            source_dir='training',
            framework_version='2.0.1',
            py_version='py310'
        )
        
        # Deploy endpoint
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name
        )
        
        return predictor


def main():
    """Example usage of AgentDistillationJob"""
    
    # Initialize job manager
    job_manager = AgentDistillationJob(region='us-east-1')
    
    # Example training job
    job_name = f"mangomas-distillation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Training data (replace with your S3 URI)
    train_data_s3 = "s3://your-bucket/mangomas/training-data/train.jsonl"
    eval_data_s3 = "s3://your-bucket/mangomas/training-data/eval.jsonl"
    
    # Hyperparameters
    hyperparameters = {
        'teacher_model_name': 'mistralai/Mistral-7B-v0.1',
        'student_model_name': 'microsoft/DialoGPT-medium',
        'num_train_epochs': 3,
        'distillation_alpha': 0.5,
        'temperature': 2.0,
        'use_lora': True,
        'use_fp16': True
    }
    
    # Create training job
    try:
        job_name = job_manager.create_training_job(
            job_name=job_name,
            teacher_model='mistralai/Mistral-7B-v0.1',
            student_model='microsoft/DialoGPT-medium',
            train_data_s3=train_data_s3,
            eval_data_s3=eval_data_s3,
            instance_type='ml.g5.2xlarge',
            hyperparameters=hyperparameters,
            use_spot_instances=True,
            max_wait_time=7200  # 2 hours
        )
        
        print(f"Training job created: {job_name}")
        
        # Monitor job
        job_manager.monitor_training_job(job_name)
        
    except Exception as e:
        print(f"Error creating training job: {e}")


if __name__ == "__main__":
    main() 