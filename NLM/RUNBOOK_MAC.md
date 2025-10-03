# Mac Development Runbook

Complete guide for running NLM distillation on macOS (Apple Silicon or Intel) for development and testing.

## Prerequisites Check

```bash
# Check Python version (3.10-3.12 required)
python3 --version

# Check Xcode CLI tools
xcode-select -p

# Check available disk space (need ~5GB)
df -h .
```

## Setup

### 1. Create Virtual Environment

```bash
cd /Users/iancruickshank/Distilled_Agents/NLM
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch, transformers, peft; print('All imports OK')"
```

### 3. Check Device Availability

```bash
python - <<'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
print(f"CPU cores: {torch.get_num_threads()}")
EOF
```

## Data Preparation

### Convert Training Data

```bash
# Convert prompt/completion JSONL to text format
python - <<'EOF'
from nlm.data import convert_jsonl_format

convert_jsonl_format(
    input_file="../product_manager_agent_real_data.jsonl",
    output_file="data/pm_agent_converted.jsonl"
)
print("Conversion complete")
EOF
```

### Verify Data

```bash
# Check line count
wc -l data/pm_agent_converted.jsonl

# Inspect first record
head -n 1 data/pm_agent_converted.jsonl | python -m json.tool
```

## Training Scenarios

### Scenario 1: Quick Smoke Test (5 minutes)

Use tiny models for fast iteration and testing.

```bash
python -m nlm.training.cli \
  --teacher-model-id "sshleifer/tiny-gpt2" \
  --student-model-id "sshleifer/tiny-gpt2" \
  --train-file "data/pm_agent_converted.jsonl" \
  --output-dir "outputs/smoke_test" \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --max-length 64 \
  --logging-steps 5 \
  --save-steps 50 \
  --no-lora \
  --no-wandb
```

**Expected**: Completes in ~5 min on M1/M2, confirms pipeline works.

### Scenario 2: Small Model Training (30 minutes)

Real models but small size for local development.

```bash
# Set models via environment
export NLM_TEACHER_MODEL_ID="gpt2"
export NLM_STUDENT_MODEL_ID="distilgpt2"

python -m nlm.training.cli \
  --config config/config.yaml \
  --train-file "data/pm_agent_converted.jsonl" \
  --output-dir "outputs/pm_agent_small" \
  --num-train-epochs 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --max-length 256 \
  --use-lora \
  --lora-rank 8
```

**Expected**: 20-30 min on Apple Silicon with MPS, produces functional model.

### Scenario 3: Debug Mode

Enable detailed logging for troubleshooting.

```bash
export NLM_LOG_LEVEL="DEBUG"

python -m nlm.training.cli \
  --teacher-model-id "sshleifer/tiny-gpt2" \
  --student-model-id "sshleifer/tiny-gpt2" \
  --train-file "data/pm_agent_converted.jsonl" \
  --output-dir "outputs/debug" \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --no-lora
```

Check logs in `training_*.log` for detailed traces.

## Testing Locally

### Run Unit Tests

```bash
# Fast tests only
pytest tests/ -v -m "not slow"

# Specific test file
pytest tests/test_config_schema.py -v

# With coverage
pytest tests/ --cov=nlm --cov-report=term-missing
```

### Run Integration Test

```bash
# Runs actual training with tiny models
pytest tests/test_integration_training.py -v -m "slow"
```

## Inference Testing

### Start Server

```bash
# Terminal 1: Start server
python -m nlm.inference.server \
  --model-dir outputs/pm_agent_small/final \
  --port 8080
```

### Test Endpoints

```bash
# Terminal 2: Test health check
curl http://localhost:8080/ping

# Test inference
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Analyze the business value of implementing a multi-agent system",
    "max_length": 128,
    "temperature": 0.7
  }' | python -m json.tool
```

## Performance Optimization

### Use MPS Acceleration

If you have Apple Silicon:

```yaml
# config/config.yaml
device_preference:
  - mps
  - cpu
```

### Reduce Memory Usage

If running out of RAM:

```bash
python -m nlm.training.cli \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 128
```

### Monitor Resource Usage

```bash
# In separate terminal
top -pid $(pgrep -f "nlm.training.cli")

# Or use Activity Monitor.app
```

## Troubleshooting

### MPS Errors

If you see MPS-related errors:

```bash
# Force CPU mode
python -m nlm.training.cli --config config/config_cpu.yaml ...
```

Create `config/config_cpu.yaml`:
```yaml
device_preference:
  - cpu
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .
```

### Slow Performance

Mac CPU training is inherently slower. Use:
- Tiny models for dev
- Reduce batch size
- Decrease max_length
- Transfer to CUDA PC for production runs

### Disk Space Issues

```bash
# Clean old outputs
rm -rf outputs/*/checkpoint-*

# Keep only final models
find outputs -type d -name "checkpoint-*" -exec rm -rf {} +
```

## Preparing for CUDA Training

### Export Configuration

```bash
# Save your working config
cp config/config.yaml config/config_production.yaml

# Edit for CUDA
vim config/config_production.yaml
```

Update for CUDA:
```yaml
teacher_model_id: "ibm-granite/granite-3.0-8b-instruct"
student_model_id: "ibm-granite/granite-3.0-2b-instruct"
per_device_train_batch_size: 8
use_fp16: true
device_preference:
  - cuda
```

### Package Data

```bash
# Create tarball for transfer to PC
tar -czf nlm_training_data.tar.gz \
  data/ \
  config/ \
  ../product_manager_agent_real_data.jsonl
```

### Transfer to PC

```bash
# Via scp
scp nlm_training_data.tar.gz user@cuda-pc:/path/to/NLM/

# Or use GitHub/cloud storage
```

## Daily Development Workflow

```bash
# 1. Activate environment
cd /Users/iancruickshank/Distilled_Agents/NLM
source .venv/bin/activate

# 2. Pull latest code
git pull

# 3. Run tests
pytest tests/ -v -m "not slow"

# 4. Make changes and test
# ... edit code ...
pytest tests/test_<module>.py -v

# 5. Quick smoke test
python -m nlm.training.cli \
  --teacher-model-id "sshleifer/tiny-gpt2" \
  --student-model-id "sshleifer/tiny-gpt2" \
  --train-file "data/pm_agent_converted.jsonl" \
  --output-dir "outputs/smoke_test" \
  --num-train-epochs 1 \
  --no-lora

# 6. Commit changes
git add .
git commit -m "feat: description of changes"
```

## Next Steps

When ready for production training:
1. Review `RUNBOOK_CUDA.md`
2. Transfer data and config to CUDA PC
3. Run production training with Granite models
4. Monitor with W&B
5. Deploy to inference endpoints

