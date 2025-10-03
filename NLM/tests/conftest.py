"""Pytest configuration and shared fixtures."""

import os
import sys
import tempfile
import json
from pathlib import Path

import pytest
import torch

# Add NLM to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_jsonl_prompt_completion(temp_dir):
    """Create sample JSONL file with prompt/completion format."""
    jsonl_file = temp_dir / "train_prompt_completion.jsonl"
    
    data = [
        {"prompt": "What is AI?", "completion": "Artificial Intelligence is..."},
        {"prompt": "Explain ML", "completion": "Machine Learning is..."},
        {"prompt": "Define NLP", "completion": "Natural Language Processing is..."}
    ]
    
    with open(jsonl_file, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    
    return jsonl_file


@pytest.fixture
def sample_jsonl_text(temp_dir):
    """Create sample JSONL file with text format."""
    jsonl_file = temp_dir / "train_text.jsonl"
    
    data = [
        {"text": "This is sample text one."},
        {"text": "This is sample text two."},
        {"text": "This is sample text three."}
    ]
    
    with open(jsonl_file, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    
    return jsonl_file


@pytest.fixture
def sample_config_yaml(temp_dir):
    """Create sample YAML configuration file."""
    config_file = temp_dir / "config.yaml"
    
    config = """
teacher_model_id: "sshleifer/tiny-gpt2"
student_model_id: "sshleifer/tiny-gpt2"
train_file: null
output_dir: "outputs/test"
num_train_epochs: 1
per_device_train_batch_size: 1
max_length: 64
lora:
  enabled: false
distillation:
  alpha: 0.5
  temperature: 2.0
"""
    
    with open(config_file, "w") as f:
        f.write(config)
    
    return config_file


@pytest.fixture
def mock_device_cuda(monkeypatch):
    """Mock CUDA availability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda x: "Mock GPU")


@pytest.fixture
def mock_device_cpu(monkeypatch):
    """Mock CPU-only environment."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

