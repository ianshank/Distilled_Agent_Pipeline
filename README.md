# Distilled Agent Pipeline

> Enterprise-grade knowledge distillation framework for training specialized AI agents from large language models. Supports local Mac development and production CUDA training with Granite-4-MoE and compatible models.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Overview

The Distilled Agent Pipeline enables organizations to create specialized, efficient AI agents through knowledge distillation—transferring knowledge from large teacher models (e.g., Granite 3.0 8B) to smaller student models (e.g., Granite 3.0 2B) while maintaining high performance.

### Key Features

- **🚀 Production-Ready**: Enterprise-grade code with comprehensive testing and validation
- **🔧 Flexible Training**: Local Mac development with MPS/CPU, production CUDA training
- **🎯 Specialized Agents**: Train domain-specific agents (Product Manager, SQE, Architect, etc.)
- **💡 Knowledge Distillation**: KL divergence-based soft target learning with configurable temperature
- **⚡ Efficient Fine-Tuning**: LoRA adapters for parameter-efficient training (0.3-1% trainable parameters)
- **🔒 Secure by Default**: Input validation, path sanitization, no hardcoded secrets
- **📊 Observable**: W&B integration, TensorBoard logs, structured logging
- **✅ Well-Tested**: 50+ unit/contract/integration tests with 80%+ coverage

## Quick Start

### Prerequisites

- Python 3.10-3.12
- **Mac**: Xcode CLI tools, 16GB+ RAM
- **CUDA PC**: NVIDIA GPU (16GB+ VRAM), CUDA 12.x + cuDNN

### Installation

```bash
# Clone repository
git clone git@github.com:ianshank/Distilled_Agent_Pipeline.git
cd Distilled_Agent_Pipeline/NLM

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Smoke Test

```bash
# Mac: 5-minute smoke test with tiny models
./quick_start.sh mac

# CUDA: 10-minute dry run
./quick_start.sh cuda
```

### Train Your First Agent

```bash
# Set model IDs (use Granite for production)
export NLM_TEACHER_MODEL_ID="ibm-granite/granite-3.0-8b-instruct"
export NLM_STUDENT_MODEL_ID="ibm-granite/granite-3.0-2b-instruct"

# Run training
python -m nlm.training.cli \
  --config config/config.yaml \
  --train-file ../data/agents/product_manager_agent_real_data.jsonl \
  --output-dir outputs/pm_agent \
  --num-train-epochs 3 \
  --use-fp16 \
  --use-lora
```

### Start Inference Server

```bash
python -m nlm.inference.server \
  --model-dir outputs/pm_agent/final \
  --port 8080

# Test
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "As a product manager, analyze...", "max_length": 256}'
```

## Project Structure

```
Distilled_Agent_Pipeline/
├── NLM/                          # Core distillation framework
│   ├── nlm/                      # Python package
│   │   ├── config/              # Configuration management
│   │   ├── data/                # Dataset loading
│   │   ├── models/              # Model loaders, device selection
│   │   ├── training/            # Training engine & CLI
│   │   └── inference/           # Inference server
│   ├── tests/                    # Comprehensive test suite
│   ├── config/                   # Configuration files
│   ├── ARCHITECTURE.md           # Detailed architecture docs
│   ├── README.md                 # Framework documentation
│   ├── RUNBOOK_MAC.md            # Mac development guide
│   └── RUNBOOK_CUDA.md           # CUDA production guide
├── data/                         # Training data
│   └── agents/                   # Agent-specific datasets
├── scripts/                      # Utility scripts
│   ├── evaluation/              # Model evaluation
│   ├── infrastructure/          # Setup & deployment
│   └── sagemaker/               # AWS SageMaker integration
└── docs/                         # Additional documentation
```

## Documentation

### Getting Started
- [NLM Framework README](NLM/README.md) - Core framework overview
- [Mac Development Runbook](NLM/RUNBOOK_MAC.md) - Local development guide
- [CUDA Production Runbook](NLM/RUNBOOK_CUDA.md) - Production training guide
- [Quick Start Script](NLM/quick_start.sh) - Automated setup

### Architecture & Design
- [Architecture Documentation](NLM/ARCHITECTURE.md) - System design and data flow
- [Implementation Summary](NLM/IMPLEMENTATION_SUMMARY.md) - Technical details

### Training Data
See [data/agents/](data/agents/) for agent-specific training datasets:
- `product_manager_agent_real_data.jsonl` - Product management scenarios
- `sqe_agent_real_data.jsonl` - Software quality engineering
- `architect_agent_real_data.jsonl` - System architecture design
- `swe_agent_real_data.jsonl` - Software engineering tasks

## Agent Types

The pipeline supports training specialized agents for different roles:

| Agent | Purpose | Training Data | Use Cases |
|-------|---------|---------------|-----------|
| **Product Manager** | Business strategy, roadmap planning | 31 examples | Feature prioritization, stakeholder management |
| **SQE** | Quality assurance, testing | 28 examples | Test planning, bug triage, quality metrics |
| **Architect** | System design, architecture | 28 examples | Architecture decisions, tech stack selection |
| **SWE** | Software development, coding | 30 examples | Code review, implementation guidance |
| **DevOps** | Infrastructure, deployment | 25 examples | CI/CD, monitoring, infrastructure as code |
| **VP Product** | Executive strategy | 22 examples | Vision, strategic planning, stakeholder alignment |

## Key Concepts

### Knowledge Distillation

Transfer knowledge from large "teacher" models to smaller "student" models:

```
Loss = (1 - α) × Task_Loss + α × Distillation_Loss

Task_Loss = CrossEntropy(Student_Logits, Labels)
Distillation_Loss = KL_Divergence(Teacher_Logits/T, Student_Logits/T) × T²
```

- **α (alpha)**: Balance between task and distillation loss (default: 0.5)
- **T (temperature)**: Softmax temperature for soft targets (default: 2.0)

### LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning that adds trainable low-rank matrices:

- **Rank (r)**: Dimensionality of low-rank matrices (default: 16)
- **Alpha**: Scaling factor (default: 32)
- **Target Modules**: Which layers to adapt (auto-detected)
- **Benefit**: Train only 0.3-1% of parameters vs full fine-tuning

### Device Selection

Automatic fallback based on availability:

1. **CUDA**: NVIDIA GPUs (fastest, production)
2. **MPS**: Apple Silicon acceleration (Mac development)
3. **CPU**: Universal fallback (slowest, always available)

## Configuration

### Priority Levels

1. **CLI Arguments** (highest)
2. **Environment Variables** (`NLM_*` prefix)
3. **YAML Configuration**
4. **Default Values** (lowest)

### Example Configuration

```yaml
# config/config.yaml
teacher_model_id: "ibm-granite/granite-3.0-8b-instruct"
student_model_id: "ibm-granite/granite-3.0-2b-instruct"

num_train_epochs: 3
per_device_train_batch_size: 4
learning_rate: 2e-5
max_length: 512

use_fp16: true  # CUDA only
device_preference: [cuda, mps, cpu]

lora:
  enabled: true
  rank: 16
  alpha: 32

distillation:
  alpha: 0.5
  temperature: 2.0
```

### Environment Variables

```bash
# Model configuration
export NLM_TEACHER_MODEL_ID="ibm-granite/granite-3.0-8b-instruct"
export NLM_STUDENT_MODEL_ID="ibm-granite/granite-3.0-2b-instruct"

# Training parameters
export NLM_NUM_TRAIN_EPOCHS="5"
export NLM_LEARNING_RATE="1e-5"

# LoRA settings (nested with double underscore)
export NLM_LORA__ENABLED="true"
export NLM_LORA__RANK="16"

# Optional: Weights & Biases tracking
export WANDB_API_KEY="your-key"
export NLM_USE_WANDB="true"
```

## Testing

```bash
# Run fast unit tests
pytest tests/ -v -m "not slow"

# Run all tests including integration
pytest tests/ -v

# Run specific test file
pytest tests/test_config_schema.py -v

# With coverage report
pytest tests/ --cov=nlm --cov-report=html
```

### Test Categories

- **Unit Tests** (40+): Config, data, models, loss computation
- **Contract Tests** (10+): API schema validation, Flask endpoints
- **Integration Tests** (2): End-to-end training with tiny models

## Performance

### Local (Mac M1/M2)
- **Tiny Models**: 5-10 minutes (smoke test)
- **Small Models** (GPT-2): 20-30 minutes (development)
- **Memory**: 8-16GB RAM sufficient

### Production (CUDA RTX 4090)
- **Granite 8B → 2B**: 4-6 hours (full training)
- **Memory**: 16-20GB VRAM with FP16 + LoRA
- **Throughput**: ~2000 tokens/second

### Optimization Tips

- **Reduce Memory**: Decrease batch size, increase gradient accumulation
- **Increase Speed**: Use FP16, optimize batch size, multi-GPU
- **Balance Quality**: Tune alpha (distillation weight) and temperature

## Security

- ✅ **No Hardcoded Secrets**: All sensitive data via environment variables
- ✅ **Input Validation**: Pydantic schemas on all external inputs
- ✅ **Path Sanitization**: Rejects writes to system directories
- ✅ **Least Privilege**: Minimal file system access, frozen teacher model
- ✅ **Secure Logging**: Secrets redacted from logs

## Troubleshooting

### Common Issues

**Out of Memory (CUDA)**
```bash
# Reduce batch size
--per-device-train-batch-size 1 \
--gradient-accumulation-steps 16

# Reduce sequence length
--max-length 256
```

**LoRA Fails**
```bash
# Disable LoRA
--no-lora
# Framework continues without LoRA if targets not found
```

**MPS Errors (Mac)**
```bash
# Force CPU mode
device_preference: [cpu]
```

**Import Errors**
```bash
# Reinstall in development mode
pip install -e .
```

See [Troubleshooting](NLM/README.md#troubleshooting) for more details.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow code quality standards (PEP 8, type hints, docstrings)
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/ -v`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards

- **Style**: PEP 8 compliance (use `black` formatter)
- **Types**: Full type hint coverage
- **Docs**: Docstrings on all public APIs
- **Tests**: Unit tests for new features
- **Security**: No hardcoded secrets, input validation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **IBM Granite**: Foundation models for knowledge distillation
- **HuggingFace**: Transformers library and model hub
- **PyTorch**: Deep learning framework
- **PEFT**: Parameter-efficient fine-tuning library

## Support

- **Issues**: [GitHub Issues](https://github.com/ianshank/Distilled_Agent_Pipeline/issues)
- **Documentation**: See `NLM/` directory for detailed docs
- **Examples**: Check `tests/` for usage examples

## Roadmap

### v1.1 (Next Release)
- [ ] Multi-GPU training support (DistributedDataParallel)
- [ ] Model quantization (INT8/INT4) for inference
- [ ] MLflow model registry integration
- [ ] GitHub Actions CI/CD pipeline

### v1.2 (Future)
- [ ] Multi-teacher distillation
- [ ] Curriculum learning support
- [ ] Active learning for data selection
- [ ] Cross-architecture distillation

## Citation

If you use this framework in your research or production systems, please cite:

```bibtex
@software{distilled_agent_pipeline,
  author = {Cruickshank, Ian},
  title = {Distilled Agent Pipeline: Enterprise Knowledge Distillation Framework},
  year = {2025},
  url = {https://github.com/ianshank/Distilled_Agent_Pipeline}
}
```

---

Ian Cruickshank
October, 2025

For detailed documentation, see the [NLM Framework README](NLM/README.md) and [Architecture Guide](NLM/ARCHITECTURE.md).
