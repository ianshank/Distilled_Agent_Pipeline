# CUDA PC Production Runbook

Complete guide for running NLM distillation on CUDA-enabled PC for production training with Granite-4-MoE models.

## Prerequisites Check

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check Python version
python3 --version

# Check available VRAM
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

**Requirements**:
- CUDA 12.x + cuDNN
- 16GB+ VRAM for Granite 8B models
- 32GB+ VRAM for full Granite distillation
- 50GB+ free disk space

## Setup

### 1. Create Virtual Environment

```bash
cd /path/to/Distilled_Agents/NLM
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install CUDA PyTorch

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("All systems operational")
EOF
```

## Environment Configuration

### Set Model IDs

```bash
# For Granite 3.0 models
export NLM_TEACHER_MODEL_ID="ibm-granite/granite-3.0-8b-instruct"
export NLM_STUDENT_MODEL_ID="ibm-granite/granite-3.0-2b-instruct"

# HuggingFace token (if using gated models)
export HF_TOKEN="your_huggingface_token"
```

### Optional: Weights & Biases

```bash
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="nlm-granite-distillation"
```

## Data Preparation

### Transfer from Mac

```bash
# Unpack data transferred from Mac
tar -xzf nlm_training_data.tar.gz

# Verify data
ls -lh data/
head -n 1 data/pm_agent_converted.jsonl | python -m json.tool
```

### Convert if Needed

```bash
python - <<'EOF'
from nlm.data import convert_jsonl_format

convert_jsonl_format(
    input_file="product_manager_agent_real_data.jsonl",
    output_file="data/pm_agent_converted.jsonl"
)
EOF
```

## Production Training

### Configuration

Create `config/config_production.yaml`:

```yaml
# Granite 3.0 Models
teacher_model_id: "ibm-granite/granite-3.0-8b-instruct"
student_model_id: "ibm-granite/granite-3.0-2b-instruct"

# Training parameters optimized for CUDA
num_train_epochs: 3
per_device_train_batch_size: 4
per_device_eval_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-5
max_length: 512

# CUDA optimizations
use_fp16: true
use_device_map: false  # Set true for multi-GPU
device_preference:
  - cuda

# LoRA for efficient fine-tuning
lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj

# Distillation
distillation:
  alpha: 0.5
  temperature: 2.0

# Monitoring
logging_steps: 50
save_steps: 500
save_total_limit: 2
eval_steps: 500
use_wandb: true

# Agent info
agent_name: "product-manager-agent"
agent_role: "Product Manager"
```

### Launch Training

```bash
python -m nlm.training.cli \
  --config config/config_production.yaml \
  --train-file data/pm_agent_converted.jsonl \
  --eval-file data/pm_agent_eval.jsonl \
  --output-dir outputs/pm_agent_prod \
  --use-wandb
```

### Monitor Training

```bash
# Terminal 1: Watch GPU usage
watch -n 1 nvidia-smi

# Terminal 2: Follow logs
tail -f training_*.log

# Terminal 3: TensorBoard
tensorboard --logdir outputs/pm_agent_prod/logs --port 6006
```

## Training Scenarios

### Scenario 1: Full Granite Distillation (8B → 2B)

**Time**: 4-6 hours on RTX 4090
**VRAM**: ~20GB

```bash
python -m nlm.training.cli \
  --config config/config_production.yaml \
  --train-file data/pm_agent_converted.jsonl \
  --output-dir outputs/granite_full \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --use-fp16 \
  --use-lora \
  --use-wandb
```

### Scenario 2: Memory-Constrained (8GB VRAM)

```bash
python -m nlm.training.cli \
  --config config/config_production.yaml \
  --train-file data/pm_agent_converted.jsonl \
  --output-dir outputs/granite_small \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 256 \
  --use-fp16 \
  --use-lora \
  --lora-rank 8
```

### Scenario 3: Multi-GPU Training

```bash
# Use all available GPUs
python -m torch.distributed.launch \
  --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) \
  -m nlm.training.cli \
  --config config/config_production.yaml \
  --train-file data/pm_agent_converted.jsonl \
  --output-dir outputs/granite_multi_gpu \
  --use-device-map
```

### Scenario 4: Dry Run Test (10 minutes)

Test full pipeline before long training:

```bash
python -m nlm.training.cli \
  --config config/config_production.yaml \
  --train-file data/pm_agent_converted.jsonl \
  --output-dir outputs/dry_run \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --max-length 128 \
  --save-steps 10 \
  --no-wandb
```

## Performance Tuning

### Optimize Batch Size

```bash
# Find optimal batch size
for bs in 1 2 4 8; do
  echo "Testing batch size: $bs"
  python -m nlm.training.cli \
    --config config/config_production.yaml \
    --train-file data/pm_agent_converted.jsonl \
    --output-dir outputs/tune_bs_${bs} \
    --num-train-epochs 1 \
    --per-device-train-batch-size $bs \
    --logging-steps 5 || break
done
```

### Mixed Precision (AMP)

Already enabled with `--use-fp16`. For more aggressive optimization:

```yaml
# In config
use_fp16: true
fp16_opt_level: "O2"  # More aggressive mixed precision
```

### Gradient Checkpointing

For very large models, enable gradient checkpointing to save VRAM:

```python
# Modify model loading in nlm/models/loaders.py
model.gradient_checkpointing_enable()
```

## Inference Deployment

### Test Trained Model

```bash
# Start inference server
python -m nlm.inference.server \
  --model-dir outputs/pm_agent_prod/final \
  --port 8080

# Test in another terminal
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "As a product manager, analyze the business value of implementing multi-agent AI systems",
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }' | python -m json.tool
```

### Production Server with Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run production server
gunicorn \
  --bind 0.0.0.0:8080 \
  --workers 2 \
  --timeout 120 \
  --worker-class sync \
  "nlm.inference.server:create_flask_app('outputs/pm_agent_prod/final')"
```

## Monitoring and Debugging

### CUDA Out of Memory

```bash
# Reduce batch size
--per-device-train-batch-size 1

# Reduce sequence length
--max-length 256

# Increase gradient accumulation
--gradient-accumulation-steps 16

# Reduce LoRA rank
--lora-rank 4
```

### Slow Training

```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Enable profiling
python -m torch.utils.bottleneck -m nlm.training.cli ...
```

### NaN Loss

```bash
# Reduce learning rate
--learning-rate 1e-5

# Clip gradients (already enabled in Trainer)
# Check data for issues
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("json", data_files={"train": "data/pm_agent_converted.jsonl"})
print(ds["train"][0])
EOF
```

## Backup and Recovery

### Save Checkpoints

```bash
# Checkpoints auto-saved to
ls -lh outputs/pm_agent_prod/checkpoint-*

# Copy to backup
rsync -avz outputs/pm_agent_prod/ /backup/nlm/pm_agent_prod/
```

### Resume Training

```bash
# Training auto-resumes from latest checkpoint
python -m nlm.training.cli \
  --config config/config_production.yaml \
  --train-file data/pm_agent_converted.jsonl \
  --output-dir outputs/pm_agent_prod  # Same directory
```

## Multi-Agent Training Pipeline

### Train All Agents

```bash
#!/bin/bash
# train_all_agents.sh

AGENTS=(
  "product_manager"
  "sqe"
  "architect"
  "swe"
  "devops"
)

for agent in "${AGENTS[@]}"; do
  echo "Training ${agent} agent..."
  
  python -m nlm.training.cli \
    --config config/config_production.yaml \
    --train-file "data/${agent}_agent_converted.jsonl" \
    --output-dir "outputs/${agent}_agent" \
    --agent-name "${agent}-agent" \
    --use-wandb
  
  if [ $? -ne 0 ]; then
    echo "Training failed for ${agent}"
    exit 1
  fi
done

echo "All agents trained successfully"
```

## Cleanup

```bash
# Remove old checkpoints, keep final models
find outputs -type d -name "checkpoint-*" -exec rm -rf {} +

# Archive completed training
tar -czf nlm_trained_models_$(date +%Y%m%d).tar.gz outputs/*/final

# Free disk space
rm -rf outputs/*/checkpoint-*
```

## Production Checklist

Before production deployment:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Dry run completes successfully
- [ ] VRAM usage monitored and acceptable
- [ ] W&B dashboards configured
- [ ] Backup strategy in place
- [ ] Inference server tested
- [ ] Model quality validated on holdout set
- [ ] Documentation updated

## Next Steps

1. Deploy trained models to inference endpoints
2. Integrate with application APIs
3. Monitor inference performance
4. Collect user feedback for next iteration
5. Plan continuous training pipeline

