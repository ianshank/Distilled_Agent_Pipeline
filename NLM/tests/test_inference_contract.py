"""
Contract tests for inference server API.

Tests Flask/SageMaker inference endpoints return schema-compliant JSON
and handle errors correctly.
"""

import pytest
import json
from unittest.mock import Mock, patch

from nlm.inference.server import InferenceRequest, InferenceResponse, create_flask_app
from pydantic import ValidationError


class TestInferenceRequest:
    """Test inference request validation."""
    
    def test_valid_request(self):
        """Test valid request passes validation."""
        request = InferenceRequest(
            prompt="Test prompt",
            max_length=256,
            temperature=0.8
        )
        
        assert request.prompt == "Test prompt"
        assert request.max_length == 256
        assert request.temperature == 0.8
    
    def test_prompt_length_validation(self):
        """Test prompt length constraints."""
        # Empty prompt
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="")
        
        # Too long prompt
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="x" * 20000)
    
    def test_prompt_sanitization(self):
        """Test prompt is sanitized."""
        request = InferenceRequest(prompt="  Test prompt  \n\n")
        assert request.prompt == "Test prompt"
    
    def test_parameter_ranges(self):
        """Test parameter validation ranges."""
        # Invalid temperature
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="test", temperature=0.0)
        
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="test", temperature=3.0)
        
        # Invalid max_length
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="test", max_length=0)
        
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="test", max_length=5000)
        
        # Invalid num_return_sequences
        with pytest.raises(ValidationError):
            InferenceRequest(prompt="test", num_return_sequences=10)


class TestInferenceResponse:
    """Test inference response validation."""
    
    def test_valid_response(self):
        """Test valid response creation."""
        response = InferenceResponse(
            responses=["Generated text"],
            input_prompt="Test prompt",
            generation_config={"temperature": 0.7}
        )
        
        assert len(response.responses) == 1
        assert response.input_prompt == "Test prompt"
        assert "temperature" in response.generation_config


class TestFlaskEndpoints:
    """Test Flask inference server endpoints."""
    
    @pytest.fixture
    def mock_model_dir(self, temp_dir):
        """Create mock model directory."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        return str(model_dir)
    
    @pytest.fixture
    def mock_inference_server(self, mock_model_dir):
        """Mock InferenceServer to avoid loading real models."""
        with patch("nlm.inference.server.InferenceServer") as MockServer:
            mock_instance = Mock()
            mock_instance.generate = Mock(return_value=InferenceResponse(
                responses=["Mocked response"],
                input_prompt="Test",
                generation_config={"temperature": 0.7}
            ))
            MockServer.return_value = mock_instance
            yield MockServer
    
    def test_ping_endpoint(self, mock_inference_server, mock_model_dir):
        """Test /ping health check endpoint."""
        app = create_flask_app(mock_model_dir)
        client = app.test_client()
        
        response = client.get("/ping")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
    
    def test_invocations_valid_request(self, mock_inference_server, mock_model_dir):
        """Test /invocations with valid request."""
        app = create_flask_app(mock_model_dir)
        client = app.test_client()
        
        request_data = {
            "prompt": "Test prompt",
            "max_length": 128,
            "temperature": 0.7
        }
        
        response = client.post(
            "/invocations",
            data=json.dumps(request_data),
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "responses" in data
        assert "input_prompt" in data
        assert "generation_config" in data
    
    def test_invocations_invalid_content_type(self, mock_inference_server, mock_model_dir):
        """Test /invocations rejects non-JSON content."""
        app = create_flask_app(mock_model_dir)
        client = app.test_client()
        
        response = client.post(
            "/invocations",
            data="not json",
            content_type="text/plain"
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_invocations_invalid_request(self, mock_inference_server, mock_model_dir):
        """Test /invocations validates request schema."""
        app = create_flask_app(mock_model_dir)
        client = app.test_client()
        
        # Missing required field
        request_data = {"max_length": 128}  # No prompt
        
        response = client.post(
            "/invocations",
            data=json.dumps(request_data),
            content_type="application/json"
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_invocations_generation_error(self, mock_model_dir):
        """Test /invocations handles generation errors."""
        with patch("nlm.inference.server.InferenceServer") as MockServer:
            mock_instance = Mock()
            mock_instance.generate = Mock(side_effect=RuntimeError("Generation failed"))
            MockServer.return_value = mock_instance
            
            app = create_flask_app(mock_model_dir)
            client = app.test_client()
            
            request_data = {"prompt": "Test"}
            
            response = client.post(
                "/invocations",
                data=json.dumps(request_data),
                content_type="application/json"
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

