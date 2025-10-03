#!/usr/bin/env python3
"""
MangoMAS Distilled Agent Inference Script
SageMaker-compatible inference script for distilled agent models
"""

import os
import json
import torch
import logging
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistilledAgentInference:
    """Inference handler for distilled agent models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = "/opt/ml/model"
        
    def model_fn(self, model_dir: str):
        """Load the model and tokenizer"""
        logger.info(f"Loading model from: {model_dir}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Check if LoRA adapter is present
            if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
                logger.info("Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(self.model, model_dir)
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def input_fn(self, request_body: str, request_content_type: str = "application/json"):
        """Parse input data"""
        if request_content_type == "application/json":
            input_data = json.loads(request_body)
            return input_data
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    
    def predict_fn(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions"""
        try:
            # Extract input parameters
            prompt = input_data.get("prompt", "")
            max_length = input_data.get("max_length", 512)
            temperature = input_data.get("temperature", 0.7)
            top_p = input_data.get("top_p", 0.9)
            top_k = input_data.get("top_k", 50)
            do_sample = input_data.get("do_sample", True)
            num_return_sequences = input_data.get("num_return_sequences", 1)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=input_data.get("repetition_penalty", 1.1)
                )
            
            # Decode outputs
            responses = []
            for output in outputs:
                # Remove input tokens from output
                response_tokens = output[inputs['input_ids'].shape[1]:]
                response_text = self.tokenizer.decode(
                    response_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                responses.append(response_text.strip())
            
            # Prepare result
            result = {
                "responses": responses,
                "input_prompt": prompt,
                "generation_config": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": do_sample,
                    "num_return_sequences": num_return_sequences
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"error": str(e)}
    
    def output_fn(self, prediction: Dict[str, Any], content_type: str = "application/json") -> str:
        """Format output"""
        if content_type == "application/json":
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")


# Global inference handler
inference_handler = DistilledAgentInference()


def model_fn(model_dir: str):
    """SageMaker model loading function"""
    return inference_handler.model_fn(model_dir)


def input_fn(request_body: str, request_content_type: str = "application/json"):
    """SageMaker input processing function"""
    return inference_handler.input_fn(request_body, request_content_type)


def predict_fn(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """SageMaker prediction function"""
    return inference_handler.predict_fn(input_data)


def output_fn(prediction: Dict[str, Any], content_type: str = "application/json") -> str:
    """SageMaker output formatting function"""
    return inference_handler.output_fn(prediction, content_type)


# Flask app for local testing
if __name__ == "__main__":
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Load model
    model_dir = os.getenv("MODEL_DIR", "/opt/ml/model")
    inference_handler.model_fn(model_dir)
    
    @app.route("/ping", methods=["GET"])
    def ping():
        """Health check endpoint"""
        return jsonify({"status": "healthy"})
    
    @app.route("/invocations", methods=["POST"])
    def invoke():
        """Inference endpoint"""
        try:
            # Get input data
            input_data = inference_handler.input_fn(
                request.get_data(as_text=True),
                request.content_type
            )
            
            # Generate prediction
            prediction = inference_handler.predict_fn(input_data)
            
            # Format output
            response = inference_handler.output_fn(prediction)
            
            return response, 200, {"Content-Type": "application/json"}
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # Run Flask app
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False) 