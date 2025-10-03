# 🚀 MangoMAS SageMaker Training Job Launchers

## Overview

Two Python scripts to launch SageMaker training jobs for MangoMAS agent distillation:

1. **`simple_launch_sagemaker.py`** - Simple, user-friendly launcher
2. **`launch_sagemaker_training_jobs.py`** - Advanced launcher with full features

## 📋 Prerequisites

### 1. AWS Setup
```bash
# Configure AWS credentials
aws configure

# Verify access
aws sts get-caller-identity
```

### 2. Deploy Infrastructure
```bash
# Deploy Terraform infrastructure first
cd infrastructure/terraform/sagemaker-jobs
terraform apply -var-file="terraform-production.tfvars"
```

### 3. Training Data
Ensure all training files exist in `training/` directory:
- `product_manager_agent.jsonl`
- `sqe_agent.jsonl` 
- `architect_agent.jsonl`
- `swe_agent.jsonl`
- `vp_product_agent.jsonl`
- `devops_agent.jsonl`
- `tools_agent.jsonl`
- `*_real_data.jsonl` files

## 🚀 Quick Start

### Option 1: Simple Launcher
```bash
cd training
python simple_launch_sagemaker.py
```

**Features:**
- ✅ Automatic role and bucket detection
- ✅ All 11 agent types configured
- ✅ Spot instances enabled (90% cost savings)
- ✅ Production hyperparameters
- ✅ Proper resource tagging

### Option 2: Advanced Launcher
```bash
cd training
python launch_sagemaker_training_jobs.py
```

**Features:**
- ✅ Terraform configuration integration
- ✅ Data validation and S3 upload
- ✅ Parallel job execution with concurrency control
- ✅ Real-time monitoring capabilities
- ✅ Comprehensive reporting and cost estimation

**Advanced options:**
```bash
# Launch with custom settings
python launch_sagemaker_training_jobs.py --max-concurrent 2 --region us-west-2

# Monitor existing jobs
python launch_sagemaker_training_jobs.py --monitor

# Use custom Terraform config
python launch_sagemaker_training_jobs.py --terraform-config ./custom_config.json
```

## 📊 What Gets Created

### Training Jobs (11 total)
- **Base Agents**: 3 epochs each
  - product-manager-agent
  - sqe-agent
  - architect-agent
  - swe-agent
  - vp-product-agent
  - devops-agent
  - tools-agent

- **Real-World Data Agents**: 4 epochs each
  - product-manager-agent-real-data
  - sqe-agent-real-data
  - architect-agent-real-data
  - swe-agent-real-data

### Configuration
- **Instance**: `ml.g4dn.xlarge` (GPU optimized)
- **Spot Training**: Enabled (90% cost savings)
- **Runtime**: 2 hours max per job
- **Storage**: 100GB per instance
- **Model**: `mistralai/Mistral-7B-v0.1`

## 💰 Cost Estimation

| Component | Spot Price | Regular Price |
|-----------|------------|---------------|
| Per Training Job | ~$0.50 | ~$5.00 |
| All 11 Jobs | ~$5.50 | ~$55.00 |
| S3 Storage | ~$0.10/month | ~$0.10/month |
| **Total** | **~$5.60** | **~$55.10** |

**💡 Spot instances provide 90% cost savings!**

## 📈 Monitoring

### AWS Console
- **SageMaker Console**: Monitor training job progress
- **CloudWatch Logs**: `/aws/sagemaker/TrainingJobs`
- **S3 Console**: View training data and model outputs

### Command Line
```bash
# Monitor with advanced launcher
python launch_sagemaker_training_jobs.py --monitor

# Check specific job
aws sagemaker describe-training-job --training-job-name <job-name>

# List all training jobs
aws sagemaker list-training-jobs --status-in-progress
```

## 📦 Model Outputs

Trained models are saved to:
```
s3://mangomas-sagemaker-production-{account-id}/models/
├── product-manager-agent/
│   └── model.tar.gz
├── sqe-agent/
│   └── model.tar.gz
└── ... (all agents)
```

### Download Models
```bash
# Download specific model
aws s3 cp s3://mangomas-sagemaker-production-{account-id}/models/product-manager-agent/model.tar.gz ./

# Download all models
aws s3 sync s3://mangomas-sagemaker-production-{account-id}/models/ ./models/
```

## 🔧 Configuration

### Hyperparameters
```python
hyperparameters = {
    'model_name_or_path': 'mistralai/Mistral-7B-v0.1',
    'num_train_epochs': '3',  # 4 for real-world data
    'per_device_train_batch_size': '2',
    'learning_rate': '2e-5',
    'fp16': 'True',  # Mixed precision
    'output_dir': '/opt/ml/model'
}
```

### Instance Configuration
- **Type**: `ml.g4dn.xlarge` (1 GPU, 4 vCPU, 16GB RAM)
- **Volume**: 100GB EBS
- **Spot**: Enabled for cost optimization
- **Max Runtime**: 7200 seconds (2 hours)

## ❗ Troubleshooting

### Common Issues

**1. Role Not Found**
```bash
# Check if role exists
aws iam get-role --role-name mangomas-sagemaker-sagemaker-execution-role

# If not, run Terraform first
cd infrastructure/terraform/sagemaker-jobs
terraform apply -var-file="terraform-production.tfvars"
```

**2. S3 Bucket Not Found**
```bash
# Check bucket exists
aws s3 ls mangomas-sagemaker-production-{account-id}

# Deploy infrastructure if missing
```

**3. Training Data Missing**
```bash
# Verify training files exist
ls -la training/*.jsonl

# Should show 11 files with training data
```

**4. Insufficient Permissions**
```bash
# Check SageMaker permissions
aws sagemaker list-training-jobs --max-results 1

# Check S3 permissions  
aws s3 ls s3://mangomas-sagemaker-production-{account-id}/
```

## 🎯 Next Steps

1. **Deploy Infrastructure**: Run Terraform deployment
2. **Launch Training**: Use one of the launcher scripts
3. **Monitor Progress**: Check SageMaker console and logs
4. **Download Models**: Retrieve trained models from S3
5. **Deploy to Production**: Create inference endpoints
6. **Integrate**: Connect trained models with MangoMAS

## 📚 Related Documentation

- [Terraform Infrastructure](../infrastructure/terraform/sagemaker-jobs/README.md)
- [Agent Distillation Guide](./README_AGENT_DISTILLATION.md)
- [SageMaker Training System](./README_SAGEMAKER_TRAINING.md)
- [Production Deployment](../docs/deployment/)

---

**🎉 Ready to train your MangoMAS agents with production-grade infrastructure!** 