"""
Inference server for distilled NLM models.

Provides Flask-based REST API compatible with SageMaker inference endpoints.
Includes input validation and error handling.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class InferenceRequest(BaseModel):
    """Validated inference request schema."""
    
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt")
    max_length: int = Field(default=512, ge=1, le=2048, description="Max generation length")
    temperature: float = Field(default=0.7, gt=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, gt=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling parameter")
    do_sample: bool = Field(default=True, description="Enable sampling")
    num_return_sequences: int = Field(default=1, ge=1, le=5, description="Number of responses")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    
    @field_validator("prompt")
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        """Sanitize prompt input."""
        # Remove null bytes and excessive whitespace
        sanitized = v.replace("\x00", "").strip()
        if not sanitized:
            raise ValueError("Prompt cannot be empty after sanitization")
        return sanitized


class InferenceResponse(BaseModel):
    """Validated inference response schema."""
    
    responses: list[str] = Field(..., description="Generated text responses")
    input_prompt: str = Field(..., description="Original input prompt")
    generation_config: Dict[str, Any] = Field(..., description="Generation parameters used")


class InferenceServer:
    """
    Inference server for distilled NLM models.
    
    Handles model loading, input validation, and text generation.
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize inference server.
        
        Args:
            model_dir: Directory containing model and tokenizer
        """
        self.model_dir = Path(model_dir)
        self.device = self._select_device()
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing inference server from {model_dir}")
        self.load_model()
    
    def _select_device(self) -> torch.device:
        """Select best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def load_model(self) -> None:
        """
        Load model and tokenizer from directory.
        
        Raises:
            FileNotFoundError: If model directory doesn't exist
            RuntimeError: If model loading fails
        """
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir),
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Check for LoRA adapter
            if PEFT_AVAILABLE and (self.model_dir / "adapter_config.json").exists():
                logger.info("Loading LoRA adapter")
                self.model = PeftModel.from_pretrained(self.model, str(self.model_dir))
            
            # Move to device
            if self.device.type != "cuda":  # cuda uses device_map
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded successfully: {param_count:,} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate text from validated request.
        
        Args:
            request: Validated inference request
        
        Returns:
            Validated inference response
        
        Raises:
            RuntimeError: If generation fails
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=request.max_length,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    do_sample=request.do_sample,
                    num_return_sequences=request.num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=request.repetition_penalty
                )
            
            # Decode outputs
            responses = []
            for output in outputs:
                # Remove input tokens from output
                response_tokens = output[inputs["input_ids"].shape[1]:]
                response_text = self.tokenizer.decode(
                    response_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                responses.append(response_text.strip())
            
            # Build response
            return InferenceResponse(
                responses=responses,
                input_prompt=request.prompt,
                generation_config={
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "do_sample": request.do_sample,
                    "num_return_sequences": request.num_return_sequences,
                    "repetition_penalty": request.repetition_penalty
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")


def create_flask_app(model_dir: str) -> Flask:
    """
    Create Flask application for inference server.
    
    Args:
        model_dir: Directory containing model and tokenizer
    
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Initialize server
    server = InferenceServer(model_dir)
    
    @app.route("/ping", methods=["GET"])
    def ping():
        """Health check endpoint."""
        return jsonify({"status": "healthy"}), 200
    
    @app.route("/invocations", methods=["POST"])
    def invoke():
        """Inference endpoint with validation."""
        try:
            # Parse request
            if request.content_type == "application/json":
                data = request.get_json()
            else:
                return jsonify({"error": "Content-Type must be application/json"}), 400
            
            # Validate request
            try:
                inference_request = InferenceRequest(**data)
            except Exception as e:
                logger.warning(f"Invalid request: {e}")
                return jsonify({"error": f"Invalid request: {e}"}), 400
            
            # Generate response
            try:
                response = server.generate(inference_request)
            except RuntimeError as e:
                logger.error(f"Generation error: {e}")
                return jsonify({"error": str(e)}), 500
            
            # Return validated response
            return jsonify(response.model_dump()), 200
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500
    
    return app


def main():
    """Main entry point for local inference server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NLM Inference Server")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.getenv("MODEL_DIR", "outputs/default/final"),
        help="Path to model directory"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8080)),
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    
    args = parser.parse_args()
    
    # Create Flask app
    app = create_flask_app(args.model_dir)
    
    # Run server
    logger.info(f"Starting inference server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

