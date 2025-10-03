# Scripts

Utility scripts for training, deployment, and infrastructure management.

## Directory Structure

- **sagemaker/** - AWS SageMaker training and deployment scripts
- **infrastructure/** - Infrastructure setup, deployment, and security
- **evaluation/** - Model training, evaluation, and inference utilities

## SageMaker Scripts

Scripts for running training jobs on AWS SageMaker:

- `launch_sagemaker_training_jobs.py` - Launch multiple training jobs
- `launch_all_agents_sagemaker.py` - Batch launch all agent training jobs
- `simple_launch_sagemaker.py` - Simple GPU-based SageMaker launcher
- `simple_launch_sagemaker_cpu.py` - Simple CPU-based SageMaker launcher
- `run_sagemaker_training.py` - Execute SageMaker training
- `sagemaker_distillation_job.py` - Configure distillation jobs

## Infrastructure Scripts

Scripts for infrastructure management and deployment:

- `setup_infrastructure.py` - Set up required infrastructure
- `verify_infrastructure.py` - Verify infrastructure configuration
- `security_scan.py` - Security scanning and validation
- `generate_deployment_summary.py` - Generate deployment reports
- `notify_failure.py` - Send failure notifications
- `package_to_onnx.py` - Package models to ONNX format

## Evaluation Scripts

Scripts for training, evaluation, and inference:

- `train_agent_skill.py` - Train specific agent skills
- `train_software_development_agent.py` - Train software development agents
- `train_distilled_adapter.py` - Train distilled model adapters
- `evaluate_agent_skill.py` - Evaluate trained agent performance
- `register_agent_skill.py` - Register agents in skill registry
- `validate_dataset.py` - Validate training datasets
- `inference.py` - Inference utilities and helpers

## Usage

Most scripts can be run directly:

```bash
# Example: Launch SageMaker training
python scripts/sagemaker/launch_sagemaker_training_jobs.py

# Example: Validate dataset
python scripts/evaluation/validate_dataset.py --data-file data/agents/architect_agent.jsonl
```

Refer to individual script documentation for specific usage instructions.
