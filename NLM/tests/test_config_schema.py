"""
Unit tests for configuration schema and loading.

Tests Pydantic validation, YAML loading, environment variable override,
and path sanitization.
"""

import os
import pytest
from pathlib import Path
from pydantic import ValidationError

from nlm.config import TrainingConfig, load_config


class TestLoRAConfig:
    """Test LoRA configuration validation."""
    
    def test_default_values(self):
        """Test LoRA defaults."""
        from nlm.config.schema import LoRAConfig
        
        config = LoRAConfig()
        assert config.enabled is True
        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
        assert "q_proj" in config.target_modules
    
    def test_rank_validation(self):
        """Test rank must be in valid range."""
        from nlm.config.schema import LoRAConfig
        
        with pytest.raises(ValidationError):
            LoRAConfig(rank=0)  # Too low
        
        with pytest.raises(ValidationError):
            LoRAConfig(rank=300)  # Too high


class TestDistillationConfig:
    """Test distillation configuration validation."""
    
    def test_default_values(self):
        """Test distillation defaults."""
        from nlm.config.schema import DistillationConfig
        
        config = DistillationConfig()
        assert config.alpha == 0.5
        assert config.temperature == 2.0
    
    def test_alpha_range(self):
        """Test alpha must be in [0, 1]."""
        from nlm.config.schema import DistillationConfig
        
        with pytest.raises(ValidationError):
            DistillationConfig(alpha=-0.1)
        
        with pytest.raises(ValidationError):
            DistillationConfig(alpha=1.5)
    
    def test_temperature_positive(self):
        """Test temperature must be positive."""
        from nlm.config.schema import DistillationConfig
        
        with pytest.raises(ValidationError):
            DistillationConfig(temperature=0.0)
        
        with pytest.raises(ValidationError):
            DistillationConfig(temperature=-1.0)


class TestTrainingConfig:
    """Test main training configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.teacher_model_id == "sshleifer/tiny-gpt2"
        assert config.student_model_id == "distilgpt2"
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 2
        assert config.max_length == 512
    
    def test_path_validation_system_dirs(self):
        """Test rejection of protected system paths."""
        with pytest.raises(ValidationError):
            TrainingConfig(output_dir="/etc/passwd")
        
        with pytest.raises(ValidationError):
            TrainingConfig(train_file="/sys/kernel/debug")
    
    def test_hyperparameter_validation(self):
        """Test hyperparameter range validation."""
        with pytest.raises(ValidationError):
            TrainingConfig(num_train_epochs=0)  # Must be >= 1
        
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-0.001)  # Must be positive
        
        with pytest.raises(ValidationError):
            TrainingConfig(max_length=0)  # Must be >= 1
    
    def test_redacted_dict(self):
        """Test configuration redaction for logging."""
        config = TrainingConfig(teacher_model_id="test-model")
        
        redacted = config.redacted_dict()
        assert isinstance(redacted, dict)
        assert redacted["teacher_model_id"] == "test-model"
    
    def test_env_variable_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("NLM_TEACHER_MODEL_ID", "env-model")
        monkeypatch.setenv("NLM_NUM_TRAIN_EPOCHS", "5")
        
        config = TrainingConfig()
        assert config.teacher_model_id == "env-model"
        assert config.num_train_epochs == 5
    
    def test_nested_env_override(self, monkeypatch):
        """Test nested configuration override via environment."""
        monkeypatch.setenv("NLM_LORA__ENABLED", "false")
        monkeypatch.setenv("NLM_LORA__RANK", "8")
        
        config = TrainingConfig()
        assert config.lora.enabled is False
        assert config.lora.rank == 8


class TestLoadConfig:
    """Test configuration loading from YAML and environment."""
    
    def test_load_from_yaml(self, sample_config_yaml):
        """Test loading configuration from YAML file."""
        config = load_config(config_path=str(sample_config_yaml))
        
        assert config.teacher_model_id == "sshleifer/tiny-gpt2"
        assert config.num_train_epochs == 1
        assert config.per_device_train_batch_size == 1
    
    def test_load_nonexistent_file(self):
        """Test error on nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            load_config(config_path="nonexistent.yaml")
    
    def test_cli_overrides(self, sample_config_yaml):
        """Test CLI overrides take precedence."""
        overrides = {
            "num_train_epochs": 10,
            "learning_rate": 1e-4
        }
        
        config = load_config(
            config_path=str(sample_config_yaml),
            overrides=overrides
        )
        
        assert config.num_train_epochs == 10  # Override
        assert config.learning_rate == 1e-4  # Override
        assert config.teacher_model_id == "sshleifer/tiny-gpt2"  # From YAML
    
    def test_env_override_priority(self, monkeypatch):
        """Test environment variables override defaults and YAML."""
        monkeypatch.setenv("NLM_NUM_TRAIN_EPOCHS", "7")
        monkeypatch.setenv("NLM_TEACHER_MODEL_ID", "env-override-model")
        
        # Load without YAML to test pure env override
        config = load_config(config_path=None)
        
        assert config.num_train_epochs == 7  # From environment
        assert config.teacher_model_id == "env-override-model"  # From environment
    
    def test_load_without_config_file(self):
        """Test loading with defaults only."""
        config = load_config()
        
        assert config.teacher_model_id == "sshleifer/tiny-gpt2"
        assert config.num_train_epochs == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

