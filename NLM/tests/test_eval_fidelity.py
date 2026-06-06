"""Unit tests for teacher-student fidelity metrics on synthetic logits."""

import pytest
import torch

from nlm.eval.fidelity import kl_fidelity, top1_agreement


class TestTop1Agreement:
    def test_identical_logits_full_agreement(self):
        logits = torch.randn(2, 5, 10)
        assert top1_agreement(logits, logits.clone()) == pytest.approx(1.0)

    def test_disagreement(self):
        student = torch.zeros(1, 1, 3)
        student[0, 0, 0] = 10.0  # argmax index 0
        teacher = torch.zeros(1, 1, 3)
        teacher[0, 0, 2] = 10.0  # argmax index 2
        assert top1_agreement(student, teacher) == 0.0

    def test_attention_mask_excludes_positions(self):
        student = torch.zeros(1, 2, 3)
        teacher = torch.zeros(1, 2, 3)
        # Position 0 agrees, position 1 disagrees.
        student[0, 0, 0] = 5.0
        teacher[0, 0, 0] = 5.0
        student[0, 1, 0] = 5.0
        teacher[0, 1, 1] = 5.0
        mask = torch.tensor([[1, 0]])  # only count position 0
        assert top1_agreement(student, teacher, mask) == pytest.approx(1.0)

    def test_zero_mask_returns_zero(self):
        logits = torch.randn(1, 2, 3)
        mask = torch.zeros(1, 2)
        assert top1_agreement(logits, logits.clone(), mask) == 0.0


class TestKLFidelity:
    def test_identical_distributions_zero_kl(self):
        logits = torch.randn(2, 4, 8)
        assert kl_fidelity(logits, logits.clone()) == pytest.approx(0.0, abs=1e-5)

    def test_divergent_distributions_positive_kl(self):
        student = torch.zeros(1, 1, 4)
        teacher = torch.zeros(1, 1, 4)
        student[0, 0, 0] = 10.0
        teacher[0, 0, 3] = 10.0
        assert kl_fidelity(student, teacher) > 0.0

    def test_non_negative(self):
        student = torch.randn(2, 3, 5)
        teacher = torch.randn(2, 3, 5)
        assert kl_fidelity(student, teacher) >= 0.0

    def test_zero_mask_returns_zero(self):
        logits = torch.randn(1, 2, 4)
        mask = torch.zeros(1, 2)
        assert kl_fidelity(logits, logits.clone() + 1.0, attention_mask=mask) == 0.0

    def test_temperature_scaling_runs(self):
        student = torch.randn(1, 2, 6)
        teacher = torch.randn(1, 2, 6)
        val = kl_fidelity(student, teacher, temperature=2.0)
        assert isinstance(val, float)
        assert val >= 0.0
