# MangoMAS SageMaker Training System

## 🎯 Overview

The MangoMAS SageMaker Training System provides automated training for specialized AI agents using real-world training data derived from actual MangoMAS application reports and patterns. This system leverages AWS SageMaker to train 11 different agent types with comprehensive, domain-specific knowledge.

## 🚀 Features

### **Multi-Agent Training**
- **11 Agent Types**: Product Manager, SQE, Architect, SWE, VP Product, DevOps, Tools
- **Real-World Data**: Training examples based on actual MangoMAS application patterns
- **Parallel Execution**: Launch multiple training jobs concurrently
- **Comprehensive Coverage**: Both basic and advanced training datasets

### **AWS Integration**
- **SageMaker Integration**: Automated training job management
- **S3 Storage**: Training data and model artifact storage
- **IAM Security**: Role-based access control
- **CloudWatch Monitoring**: Training progress and performance tracking

### **Training Data Quality**
- **Real Application Patterns**: Based on actual MangoMAS reports and implementations
- **Domain Expertise**: Specialized knowledge for each agent type
- **Production Scenarios**: Real-world use cases and configurations
- **Comprehensive Examples**: 330+ total training examples across all agents

## 📊 Training Data Overview

### **Real-World Training Datasets**

| Agent Type | Training File | Examples | Focus Area |
|------------|---------------|----------|------------|
| Product Manager | `product_manager_agent_real_data.jsonl` | 30 | Business strategy, ROI analysis, product roadmaps |
| SQE | `sqe_agent_real_data.jsonl` | 30 | Testing frameworks, security validation, performance testing |
| Architect | `architect_agent_real_data.jsonl` | 30 | System design, AWS integration, scalability patterns |
| SWE | `swe_agent_real_data.jsonl` | 30 | Code implementation, deployment, monitoring systems |
| VP Product | `vp_product_agent.jsonl` | 30 | Strategic leadership, market analysis, executive decisions |
| DevOps | `devops_agent.jsonl` | 30 | Infrastructure automation, CI/CD, cloud operations |
| Tools | `tools_agent.jsonl` | 50 | Tool integration, utility development, automation |

### **Basic Training Datasets**
- `product_manager_agent.jsonl` (30 examples)
- `sqe_agent.jsonl` (30 examples)
- `architect_agent.jsonl` (30 examples)
- `swe_agent.jsonl` (30 examples)

## 🛠️ Installation & Setup

### **Prerequisites**
```bash
# Install required packages
pip install sagemaker boto3 transformers datasets torch

# Configure AWS credentials
aws configure
# or set environment variables:
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

### **SageMaker Role Setup**
```bash
# Create SageMaker execution role (if needed)
aws iam create-role --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach required policies
aws iam attach-role-policy --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

## 🚀 Quick Start

### **1. Simple Training Launch**
```bash
cd training
python run_sagemaker_training.py
```

### **2. Custom Configuration**
```python
from launch_all_agents_sagemaker import MangoMASSageMakerLauncher
import asyncio

# Create launcher with custom configuration
launcher = MangoMASSageMakerLauncher(
    region='us-east-1',
    role_arn='arn:aws:iam::123456789012:role/SageMakerExecutionRole'
)

# Launch training jobs
results = asyncio.run(launcher.launch_all_jobs(
    parallel=True,
    max_concurrent=3
))
```

### **3. Individual Agent Training**
```python
from launch_all_agents_sagemaker import MangoMASSageMakerLauncher, TrainingJobConfig

launcher = MangoMASSageMakerLauncher()

# Train specific agent
config = TrainingJobConfig(
    agent_name="product_manager",
    training_file="product_manager_agent_real_data.jsonl",
    epochs=3,
    batch_size=2
)

result = asyncio.run(launcher.launch_training_job(config, s3_uri))
```

## 📚 Usage

### **Command Line Usage**
```bash
# Basic usage
cd training
python run_sagemaker_training.py

# With environment variables
export AWS_REGION=us-east-1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole
python run_sagemaker_training.py
```

### **Programmatic Usage**
```python
import asyncio
from launch_all_agents_sagemaker import MangoMASSageMakerLauncher

# Initialize launcher
launcher = MangoMASSageMakerLauncher(
    region='us-east-1',
    role_arn='arn:aws:iam::123456789012:role/SageMakerExecutionRole'
)

# Launch all training jobs
async def main():
    results = await launcher.launch_all_jobs(
        parallel=True,
        max_concurrent=3
    )
    return results

# Run training
results = asyncio.run(main())
```

### **Monitoring Training Jobs**
```python
# Check training status
launcher = MangoMASSageMakerLauncher()
status = launcher.get_training_status()

# Generate summary report
summary = launcher.generate_summary_report()
print(summary)

# Save results
launcher.save_results('training_results.json')
```

### **Custom Training Configuration**
```python
from launch_all_agents_sagemaker import TrainingJobConfig

# Create custom configuration
config = TrainingJobConfig(
    agent_name="custom_agent",
    training_file="custom_training_data.jsonl",
    model_name="microsoft/DialoGPT-medium",
    epochs=5,
    batch_size=4,
    learning_rate=1e-4,
    instance_type="ml.g4dn.2xlarge"
)

# Launch training with custom config
result = await launcher.launch_training_job(config, s3_uri)
```

## 📋 Training Configuration

### **Default Hyperparameters**
```python
TrainingJobConfig(
    model_name='mistralai/Mistral-7B-v0.1',
    instance_type='ml.g4dn.xlarge',
    epochs=3,
    batch_size=2,
    learning_rate=2e-5,
    max_length=512
)
```

### **Instance Types**
- **ml.g4dn.xlarge**: GPU instance for faster training
- **ml.m5.large**: CPU instance for cost optimization
- **ml.p3.2xlarge**: High-performance GPU for complex models

### **Training Optimization**
- **Mixed Precision**: FP16 training for faster convergence
- **Gradient Accumulation**: Effective batch size optimization
- **Learning Rate Scheduling**: Warmup and decay strategies
- **Early Stopping**: Prevent overfitting with validation monitoring

## 📊 Monitoring & Results

### **Training Progress**
```bash
# Monitor jobs in SageMaker console
aws sagemaker list-training-jobs --status-in-progress

# Check specific job status
aws sagemaker describe-training-job --training-job-name mangomas-product-manager-20250130-143022
```

### **Results Analysis**
```python
# Load training results
with open('training_results.json', 'r') as f:
    results = json.load(f)

print(f"Success Rate: {results['summary']['success_rate']:.1%}")
print(f"Total Jobs: {results['summary']['total_jobs']}")
```

### **Model Artifacts**
- **S3 Location**: `s3://mangomas-training-data/models/{agent_name}/`
- **Model Files**: Trained model weights and configuration
- **Training Logs**: Detailed training metrics and logs
- **Evaluation Results**: Performance metrics and validation scores

## 🔧 Advanced Configuration

### **Custom Training Scripts**
```python
# Modify training script for custom requirements
estimator = HuggingFace(
    entry_point='custom_training_script.py',
    source_dir='training',
    hyperparameters={
        'custom_param': 'value',
        'model_name_or_path': 'custom-model',
        'num_train_epochs': 5
    }
)
```

### **Multi-GPU Training**
```python
# Configure for multi-GPU training
estimator = HuggingFace(
    instance_type='ml.p3.8xlarge',  # 4 GPUs
    instance_count=1,
    hyperparameters={
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 2
    }
)
```

### **Custom Evaluation**
```python
# Add custom evaluation metrics
hyperparameters = {
    'evaluation_strategy': 'steps',
    'eval_steps': 500,
    'metric_for_best_model': 'custom_metric',
    'load_best_model_at_end': True
}
```

## 📈 Performance Optimization

### **Training Speed**
- **Mixed Precision**: 2x faster training with FP16
- **Gradient Accumulation**: Larger effective batch sizes
- **Data Loading**: Optimized data pipeline with multiple workers
- **Model Optimization**: Quantization and pruning techniques

### **Cost Optimization**
- **Spot Instances**: Up to 90% cost savings
- **Auto-scaling**: Dynamic resource allocation
- **Early Stopping**: Prevent unnecessary training time
- **Model Compression**: Smaller models for faster inference

### **Quality Assurance**
- **Validation Metrics**: Comprehensive evaluation criteria
- **Cross-validation**: Robust model performance assessment
- **A/B Testing**: Model comparison and selection
- **Continuous Monitoring**: Performance tracking over time

## 🔒 Security & Compliance

### **Data Security**
- **Encryption**: S3 server-side encryption for training data
- **Access Control**: IAM roles with least privilege access
- **Network Security**: VPC isolation for training jobs
- **Audit Logging**: Comprehensive access and usage logs

### **Model Security**
- **Model Encryption**: Encrypted model artifacts
- **Access Control**: Role-based model access
- **Version Control**: Model versioning and rollback
- **Compliance**: SOC 2, GDPR, HIPAA compliance support

## 🚀 Deployment

### **Model Deployment**
```python
# Deploy trained model to SageMaker endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=f'mangomas-{agent_name}-endpoint'
)
```

### **Inference Pipeline**
```python
# Create inference pipeline
from sagemaker.pipeline import Pipeline

pipeline = Pipeline(
    name='mangomas-inference-pipeline',
    steps=[preprocessing_step, inference_step, postprocessing_step]
)
```

### **Production Monitoring**
- **CloudWatch Metrics**: Real-time performance monitoring
- **Custom Dashboards**: Agent-specific performance tracking
- **Alerting**: Automated notifications for issues
- **Logging**: Comprehensive inference logging

## 📚 Examples

### **Complete Training Workflow**
```python
import asyncio
from launch_all_agents_sagemaker import MangoMASSageMakerLauncher

async def train_all_agents():
    # Initialize launcher
    launcher = MangoMASSageMakerLauncher()
    
    # Launch training jobs
    results = await launcher.launch_all_jobs(
        parallel=True,
        max_concurrent=3
    )
    
    # Generate report
    summary = launcher.generate_summary_report()
    print(summary)
    
    # Save results
    launcher.save_results()
    
    return results

# Run training
results = asyncio.run(train_all_agents())
```

### **Custom Agent Training**
```python
from launch_all_agents_sagemaker import TrainingJobConfig

# Create custom training configuration
custom_config = TrainingJobConfig(
    agent_name="custom_agent",
    training_file="custom_training_data.jsonl",
    model_name="microsoft/DialoGPT-medium",
    epochs=5,
    batch_size=4,
    learning_rate=1e-4
)

# Train custom agent
result = await launcher.launch_training_job(custom_config, s3_uri)
```

## 🐛 Troubleshooting

### **Common Issues**

#### **AWS Credentials Error**
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Configure credentials
aws configure
```

#### **SageMaker Role Error**
```bash
# Check role permissions
aws iam get-role --role-name SageMakerExecutionRole

# Create role if missing
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json
```

#### **Training Job Failure**
```bash
# Check training job logs
aws sagemaker describe-training-job --training-job-name <job-name>

# Download logs
aws s3 cp s3://<bucket>/<job-name>/output/logs/ ./logs/
```

### **Performance Issues**
- **Memory Errors**: Reduce batch size or use larger instance
- **Slow Training**: Enable mixed precision or use GPU instances
- **Cost Issues**: Use spot instances or optimize hyperparameters

## 📞 Support

### **Documentation**
- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [MangoMAS Architecture](docs/architecture/)

### **Monitoring**
- **SageMaker Console**: Real-time training job monitoring
- **CloudWatch**: Performance metrics and logs
- **S3 Console**: Model artifacts and training data

### **Contact**
- **Issues**: Create GitHub issue for bugs or feature requests
- **Questions**: Check documentation or create discussion
- **Support**: Contact team for enterprise support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎯 Ready to train your MangoMAS agents with real-world data!**

Run `python run_sagemaker_training.py` to start training all 11 agent types with comprehensive, domain-specific knowledge derived from actual MangoMAS application patterns. 