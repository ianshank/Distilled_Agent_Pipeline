#!/bin/bash
# Quick start script for NLM distillation training
# Usage: ./quick_start.sh [mac|cuda]

set -e

MODE="${1:-mac}"

echo "NLM Distillation Quick Start"
echo "============================="
echo "Mode: $MODE"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt > /dev/null

# Run based on mode
if [ "$MODE" = "mac" ]; then
    echo ""
    echo "Running Mac smoke test with tiny models..."
    echo "This will take ~5 minutes"
    echo ""
    
    # Create test data if needed
    if [ ! -f "../product_manager_agent_real_data.jsonl" ]; then
        echo "Error: Training data not found"
        echo "Expected: ../product_manager_agent_real_data.jsonl"
        exit 1
    fi
    
    python -m nlm.training.cli \
        --teacher-model-id "sshleifer/tiny-gpt2" \
        --student-model-id "sshleifer/tiny-gpt2" \
        --train-file "../product_manager_agent_real_data.jsonl" \
        --output-dir "outputs/quick_start" \
        --num-train-epochs 1 \
        --per-device-train-batch-size 1 \
        --max-length 64 \
        --logging-steps 5 \
        --save-steps 50 \
        --no-lora \
        --no-wandb
    
    echo ""
    echo "Training complete!"
    echo "Model saved to: outputs/quick_start/final"
    echo ""
    echo "Next steps:"
    echo "  1. Review logs: ls -lh training_*.log"
    echo "  2. Test inference: python -m nlm.inference.server --model-dir outputs/quick_start/final"
    echo "  3. Run full tests: pytest tests/ -v"
    echo "  4. See RUNBOOK_MAC.md for more options"

elif [ "$MODE" = "cuda" ]; then
    echo ""
    echo "Running CUDA dry run test..."
    echo "This will take ~10 minutes"
    echo ""
    
    # Check CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. CUDA may not be available."
    else
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi
    
    # Create test data if needed
    if [ ! -f "../product_manager_agent_real_data.jsonl" ]; then
        echo "Error: Training data not found"
        exit 1
    fi
    
    python -m nlm.training.cli \
        --config config/config.yaml \
        --train-file "../product_manager_agent_real_data.jsonl" \
        --output-dir "outputs/cuda_test" \
        --num-train-epochs 1 \
        --per-device-train-batch-size 2 \
        --max-length 128 \
        --use-fp16 \
        --use-lora \
        --lora-rank 8 \
        --no-wandb
    
    echo ""
    echo "Dry run complete!"
    echo "Model saved to: outputs/cuda_test/final"
    echo ""
    echo "Next steps:"
    echo "  1. Review RUNBOOK_CUDA.md for production training"
    echo "  2. Set Granite model IDs: export NLM_TEACHER_MODEL_ID=..."
    echo "  3. Enable W&B: export WANDB_API_KEY=..."
    echo "  4. Run production training with full config"

else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: $0 [mac|cuda]"
    exit 1
fi

