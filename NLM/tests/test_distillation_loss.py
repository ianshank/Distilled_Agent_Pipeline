"""
Unit tests for distillation loss computation.

Tests KL divergence + cross-entropy composition, alpha/temperature behavior,
and edge cases.
"""

import pytest
import torch

from nlm.training.trainer import compute_distillation_loss


class TestDistillationLoss:
    """Test distillation loss computation."""
    
    @pytest.fixture
    def sample_logits(self):
        """Create sample student and teacher logits."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        return student_logits, teacher_logits, labels
    
    def test_pure_task_loss(self, sample_logits):
        """Test alpha=0 returns only task loss."""
        student_logits, teacher_logits, labels = sample_logits
        
        loss_dict = compute_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            alpha=0.0,
            temperature=2.0
        )
        
        assert "total_loss" in loss_dict
        assert "task_loss" in loss_dict
        assert "distillation_loss" in loss_dict
        
        # With alpha=0, total_loss should equal task_loss
        assert torch.isclose(loss_dict["total_loss"], loss_dict["task_loss"])
        assert loss_dict["distillation_loss"].item() == 0.0
    
    def test_pure_distillation_loss(self, sample_logits):
        """Test alpha=1 returns only distillation loss."""
        student_logits, teacher_logits, labels = sample_logits
        
        loss_dict = compute_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            alpha=1.0,
            temperature=2.0
        )
        
        # With alpha=1, total_loss should equal distillation_loss
        assert torch.isclose(loss_dict["total_loss"], loss_dict["distillation_loss"])
    
    def test_mixed_loss(self, sample_logits):
        """Test alpha=0.5 gives balanced loss."""
        student_logits, teacher_logits, labels = sample_logits
        
        loss_dict = compute_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            alpha=0.5,
            temperature=2.0
        )
        
        task_loss = loss_dict["task_loss"].item()
        distill_loss = loss_dict["distillation_loss"].item()
        total_loss = loss_dict["total_loss"].item()
        
        # Total should be weighted average
        expected = 0.5 * task_loss + 0.5 * distill_loss
        assert abs(total_loss - expected) < 1e-5
    
    def test_temperature_scaling(self, sample_logits):
        """Test temperature affects distillation loss magnitude."""
        student_logits, teacher_logits, labels = sample_logits
        
        loss_low_temp = compute_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            alpha=1.0,
            temperature=1.0
        )
        
        loss_high_temp = compute_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            alpha=1.0,
            temperature=3.0
        )
        
        # Higher temperature should scale loss (due to T^2 factor)
        assert loss_high_temp["distillation_loss"] != loss_low_temp["distillation_loss"]
    
    def test_ignore_index_handling(self):
        """Test -100 labels are ignored in task loss."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)  # All ignored
        
        loss_dict = compute_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            alpha=0.0,  # Pure task loss
            temperature=2.0
        )
        
        # Task loss should still compute without error
        assert not torch.isnan(loss_dict["task_loss"])
        assert not torch.isinf(loss_dict["task_loss"])
    
    def test_loss_shapes(self, sample_logits):
        """Test loss outputs are scalar tensors."""
        student_logits, teacher_logits, labels = sample_logits
        
        loss_dict = compute_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            alpha=0.5,
            temperature=2.0
        )
        
        assert loss_dict["total_loss"].ndim == 0  # Scalar
        assert loss_dict["task_loss"].ndim == 0
        assert loss_dict["distillation_loss"].ndim == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

