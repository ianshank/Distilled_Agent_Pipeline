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

- `security_scan.py` - Security scanning and validation
- `generate_deployment_summary.py` - Generate deployment reports
- `notify_failure.py` - Send failure notifications

## Evaluation Scripts

Legacy standalone SageMaker entrypoints (kept for backwards compatibility):

- `train_distilled_adapter.py` - Standalone SageMaker distillation trainer
- `inference.py` - Standalone SageMaker inference handler

> **Note:** The maintained training, inference, evaluation, and dataset-validation
> functionality now lives in the `nlm` package (`NLM/nlm/`). Prefer:
> - Training: `nlm-train` / `python -m nlm.training.cli`
> - Inference: `nlm-serve` / `python -m nlm.inference.server`
> - Agent evaluation: `nlm-eval` / `python -m nlm.eval.cli`
> - Dataset validation: `nlm-validate-data` / `python -m nlm.data.validation`
>
> Several broken legacy scripts that imported a non-existent
> `agents.automated_training_system` module (agent-skill training/evaluation/
> registration and infrastructure setup/verify/ONNX packaging) were removed
> during a gap-analysis cleanup.

## Usage

Most scripts can be run directly:

```bash
# Example: Launch SageMaker training
python scripts/sagemaker/launch_sagemaker_training_jobs.py

# Example: Validate a dataset (now provided by the nlm package)
nlm-validate-data --path data/agents/architect_agent.jsonl --min-samples 5
```

Refer to individual script documentation for specific usage instructions.
