"""
Unit tests for LoRA target module detection and guarding.

Tests automatic detection of valid LoRA target modules across
different model architectures.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from nlm.models.loaders import detect_lora_target_modules, setup_lora_adapter


class TestLoRATargetDetection:
    """Test LoRA target module detection."""
    
    def test_detect_gpt2_modules(self):
        """Test detection of LoRA targets in GPT-2 model."""
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        
        targets = detect_lora_target_modules(model)
        
        # GPT-2 uses c_attn, c_proj
        assert len(targets) > 0
        assert any(t in ["c_attn", "c_proj"] for t in targets)
    
    def test_empty_model_no_crash(self):
        """Test graceful handling of model with no standard targets."""
        # Create minimal model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        
        targets = detect_lora_target_modules(model)
        
        # Should return empty list, not crash
        assert isinstance(targets, list)


class TestLoRASetup:
    """Test LoRA adapter setup with guarding."""
    
    def test_setup_with_auto_detection(self):
        """Test LoRA setup with automatic target detection."""
        pytest.importorskip("peft")  # Skip if peft not installed
        
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        original_param_count = sum(p.numel() for p in model.parameters())
        
        lora_model = setup_lora_adapter(
            model,
            rank=8,
            alpha=16,
            dropout=0.1,
            target_modules=None  # Auto-detect
        )
        
        # Check LoRA was applied
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        assert trainable_params > 0
        assert trainable_params < original_param_count
    
    def test_setup_with_explicit_targets(self):
        """Test LoRA setup with explicit target modules."""
        pytest.importorskip("peft")
        
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        
        lora_model = setup_lora_adapter(
            model,
            rank=8,
            alpha=16,
            dropout=0.1,
            target_modules=["c_attn", "c_proj"]
        )
        
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        assert trainable_params > 0
    
    def test_invalid_targets_fallback(self):
        """Test fallback when invalid targets provided."""
        pytest.importorskip("peft")
        
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        
        # Provide nonexistent target modules
        lora_model = setup_lora_adapter(
            model,
            rank=8,
            alpha=16,
            target_modules=["nonexistent_module", "fake_layer"]
        )
        
        # Should return original model without LoRA
        assert lora_model is model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

