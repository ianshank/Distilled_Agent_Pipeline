"""
Integration tests for end-to-end training.

Tests complete training pipeline with tiny models to verify
all components work together correctly.
"""

import pytest
import json
from pathlib import Path

from nlm.config import TrainingConfig
from nlm.training.cli import train


class TestEndToEndTraining:
    """Integration tests for complete training pipeline."""
    
    @pytest.fixture
    def minimal_config(self, temp_dir, sample_jsonl_text):
        """Create minimal config for fast training."""
        config = TrainingConfig(
            teacher_model_id="sshleifer/tiny-gpt2",
            student_model_id="sshleifer/tiny-gpt2",
            train_file=str(sample_jsonl_text),
            output_dir=str(temp_dir / "outputs"),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_length=32,
            logging_steps=1,
            save_steps=10,
            save_total_limit=1,
            use_fp16=False,
            use_wandb=False
        )
        config.lora.enabled = False  # Disable LoRA for speed
        config.distillation.alpha = 0.5
        
        return config
    
    @pytest.mark.slow
    def test_full_training_pipeline(self, minimal_config):
        """Test complete training pipeline with tiny model."""
        # This test is marked slow as it does actual model training
        pytest.importorskip("transformers")
        
        try:
            train(minimal_config)
        except Exception as e:
            pytest.fail(f"Training failed: {e}")
        
        # Verify outputs
        output_dir = Path(minimal_config.output_dir) / "final"
        assert output_dir.exists()
        
        # Check model files
        assert (output_dir / "config.json").exists()
        assert (output_dir / "training_metadata.json").exists()
        
        # Verify metadata
        with open(output_dir / "training_metadata.json") as f:
            metadata = json.load(f)
        
        assert "config" in metadata
        assert "training_completed_at" in metadata
        assert metadata["config"]["num_train_epochs"] == 1
    
    @pytest.mark.slow
    def test_training_with_lora(self, minimal_config):
        """Test training pipeline with LoRA enabled."""
        pytest.importorskip("peft")
        
        minimal_config.lora.enabled = True
        minimal_config.lora.rank = 4  # Very small for speed
        
        try:
            train(minimal_config)
        except Exception as e:
            pytest.fail(f"LoRA training failed: {e}")
        
        output_dir = Path(minimal_config.output_dir) / "final"
        assert output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])

