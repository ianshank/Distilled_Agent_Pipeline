# NLM - Neural Language Model Distillation Framework

Enterprise-grade knowledge distillation framework for Granite-4-MoE and compatible language models. Supports local Mac development (MPS/CPU) and production CUDA training with comprehensive testing and observability.

## Features

- **Flexible Device Support**: Automatic device selection (CUDA → MPS → CPU)
- **Safe LoRA Integration**: Auto-detection of target modules with fallback
- **Knowledge Distillation**: Configurable teacher-student training with KL divergence
- **Production Ready**: Pydantic validation, structured logging, comprehensive tests
- **Cloud Compatible**: SageMaker integration support
- **Secure by Default**: No hardcoded secrets, input validation, path sanitization

## Architecture

```
NLM/
├── config/                    # Configuration files
│   └── config.yaml           # Default training configuration
├── nlm/                      # Main package
│   ├── config/              # Configuration schema and loaders
│   ├── data/                # Dataset loading and preprocessing
│   ├── models/              # Model loaders and device selection
│   ├── training/            # Training logic and CLI
│   └── inference/           # Inference server
├── tests/                    # Comprehensive test suite
├── outputs/                  # Training outputs (gitignored)
└── requirements.txt         # Python dependencies
```

## Prerequisites

- **Python**: 3.10-3.12
- **Mac Development**: Xcode CLI tools, optional MPS support
- **CUDA Training**: CUDA 12.x + cuDNN, PyTorch CUDA build
- **Models**: HuggingFace account for model access
- **Optional**: W&B account for experiment tracking

## Installation

### 1. Clone and Setup Virtual Environment

```bash
cd /Users/iancruickshank/Distilled_Agents/NLM
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Base installation
pip install -r requirements.txt

# For CUDA (on PC)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Quick Start

### Local Mac Training (Development)

```bash
# Set environment variables
export NLM_TEACHER_MODEL_ID="sshleifer/tiny-gpt2"
export NLM_STUDENT_MODEL_ID="distilgpt2"

# Run training with tiny models
python -m nlm.training.cli \
  --config config/config.yaml \
  --train-file ../product_manager_agent_real_data.jsonl \
  --output-dir outputs/pm_agent_dev \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --no-lora \
  --no-wandb
```

### CUDA PC Training (Production)

```bash
# Set Granite-4-MoE models via environment
export NLM_TEACHER_MODEL_ID="ibm-granite/granite-3.0-8b-instruct"
export NLM_STUDENT_MODEL_ID="ibm-granite/granite-3.0-2b-instruct"

# Run production training
python -m nlm.training.cli \
  --config config/config.yaml \
  --train-file data/product_manager_agent_real_data.jsonl \
  --output-dir outputs/pm_agent_prod \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4 \
  --use-fp16 \
  --use-lora \
  --lora-rank 16
```

### Inference Server

```bash
# Start local inference server
python -m nlm.inference.server \
  --model-dir outputs/pm_agent_dev/final \
  --port 8080

# Test inference
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze the business value of AI agents", "max_length": 256}'
```

## Configuration

### Environment Variables

All config can be overridden via environment variables with `NLM_` prefix:

```bash
export NLM_TEACHER_MODEL_ID="your-model"
export NLM_NUM_TRAIN_EPOCHS="5"
export NLM_LORA__ENABLED="true"
export NLM_LORA__RANK="16"
```

### YAML Configuration

Edit `config/config.yaml` for persistent settings:

```yaml
teacher_model_id: "ibm-granite/granite-3.0-8b-instruct"
student_model_id: "ibm-granite/granite-3.0-2b-instruct"
num_train_epochs: 3
lora:
  enabled: true
  rank: 16
distillation:
  alpha: 0.5
  temperature: 2.0
```

### CLI Overrides

CLI flags take highest priority:

```bash
python -m nlm.training.cli \
  --teacher-model-id "custom-teacher" \
  --num-train-epochs 5 \
  --use-lora
```

## Testing

### Run All Tests

```bash
# Fast tests only (unit + contract)
pytest -v

# Include slow integration tests
pytest -v -m "slow"

# With coverage
pytest --cov=nlm --cov-report=html
```

### Test Categories

- **Unit Tests**: Config, data, models, loss computation
- **Contract Tests**: API schema validation
- **Integration Tests**: End-to-end training pipeline

## Data Format

### Input JSONL Format

```jsonl
{"prompt": "Question or task", "completion": "Expected response"}
{"prompt": "Another question", "completion": "Another response"}
```

Or pre-converted text format:

```jsonl
{"text": "Combined prompt and completion text"}
```

### Convert Format

```python
from nlm.data import convert_jsonl_format

convert_jsonl_format(
    input_file="data/raw.jsonl",
    output_file="data/converted.jsonl"
)
```

## Monitoring

### Weights & Biases

```bash
# Enable W&B tracking
export WANDB_API_KEY="your-key"
python -m nlm.training.cli --use-wandb ...
```

### Logs

Training logs are saved to:
- Console output
- `training_YYYYMMDD_HHMMSS.log`
- `outputs/<agent>/logs/` (TensorBoard)

## Troubleshooting

### LoRA Target Module Errors

If LoRA fails with "no valid targets":

```bash
# Disable LoRA temporarily
python -m nlm.training.cli --no-lora ...
```

The framework auto-detects valid targets; if none found, training continues without LoRA.

### Out of Memory

Reduce batch size and enable gradient accumulation:

```bash
python -m nlm.training.cli \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

### MPS Errors on Mac

Fall back to CPU if MPS has issues:

```yaml
device_preference:
  - cpu
```

## Development

### Code Quality

```bash
# Format code
black nlm/ tests/

# Lint
flake8 nlm/ tests/

# Type check
mypy nlm/
```

### Adding New Tests

1. Create `tests/test_<module>.py`
2. Follow existing patterns (unit/contract/integration)
3. Use fixtures from `conftest.py`
4. Run tests: `pytest tests/test_<module>.py -v`

## Security

- **No hardcoded secrets**: Use environment variables
- **Input validation**: Pydantic schemas on all external inputs
- **Path sanitization**: Rejects system directory writes
- **Dependency scanning**: Run `pip-audit` regularly

## License

See parent repository LICENSE file.

## Support

For issues, feature requests, or questions:
1. Check existing tests for usage examples
2. Review configuration schema in `nlm/config/schema.py`
3. Enable debug logging: `export NLM_LOG_LEVEL=DEBUG`

