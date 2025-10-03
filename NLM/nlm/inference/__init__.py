"""Inference server for distilled NLM models."""

from nlm.inference.server import InferenceServer, create_flask_app

__all__ = ["InferenceServer", "create_flask_app"]

