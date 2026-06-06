"""
Agent evaluation harness for distilled NLM models.

Provides a reproducible way to score a trained (distilled) student model on a
held-out behavioral benchmark using reference-based generation metrics
(exact match, token F1, ROUGE-L), keyword/behavioral checks, and optional
teacher-student fidelity metrics (top-1 agreement, KL divergence).

Public API:
    load_benchmark        -- read a JSONL benchmark into EvalCase objects
    evaluate              -- run a Generator over cases and score the outputs
    EvalCase, CaseResult, EvalReport -- result data models
    ModelGenerator        -- Generator backed by a local model directory
    DEFAULT_METRICS       -- metric names used when none are specified
    write_json_report, write_html_report -- report serialization
"""

from nlm.eval.benchmark import (
    CaseResult,
    EvalCase,
    EvalReport,
    load_benchmark,
)
from nlm.eval.metrics import (
    DEFAULT_METRICS,
    METRIC_FUNCTIONS,
    exact_match,
    keyword_recall,
    normalize_text,
    rouge_l,
    token_f1,
)
from nlm.eval.report import render_html_report, write_html_report, write_json_report
from nlm.eval.runner import Generator, ModelGenerator, evaluate

__all__ = [
    "EvalCase",
    "CaseResult",
    "EvalReport",
    "load_benchmark",
    "DEFAULT_METRICS",
    "METRIC_FUNCTIONS",
    "exact_match",
    "token_f1",
    "rouge_l",
    "keyword_recall",
    "normalize_text",
    "Generator",
    "ModelGenerator",
    "evaluate",
    "write_json_report",
    "write_html_report",
    "render_html_report",
]
