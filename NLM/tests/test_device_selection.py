"""
Unit tests for device selection logic.

Tests CUDA, MPS, and CPU device selection with mocked availability.
"""

import pytest
import torch

from nlm.models import select_device


class TestDeviceSelection:
    """Test device selection with different hardware availability."""
    
    def test_cuda_priority(self, mock_device_cuda):
        """Test CUDA selected when available."""
        device = select_device(preference=["cuda", "mps", "cpu"])
        
        assert device.type == "cuda"
    
    def test_mps_fallback(self, monkeypatch):
        """Test MPS selected when CUDA unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        
        # Mock MPS availability
        if not hasattr(torch.backends, "mps"):
            pytest.skip("MPS not available in this PyTorch version")
        
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        
        device = select_device(preference=["cuda", "mps", "cpu"])
        
        assert device.type == "mps"
    
    def test_cpu_fallback(self, mock_device_cpu):
        """Test CPU selected when no accelerators available."""
        device = select_device(preference=["cuda", "mps", "cpu"])
        
        assert device.type == "cpu"
    
    def test_custom_preference_order(self, mock_device_cpu):
        """Test custom device preference order."""
        device = select_device(preference=["mps", "cuda", "cpu"])
        
        # Should fall back to CPU since nothing available
        assert device.type == "cpu"
    
    def test_default_preference(self, mock_device_cpu):
        """Test default preference order."""
        device = select_device()  # No preference specified
        
        assert device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

