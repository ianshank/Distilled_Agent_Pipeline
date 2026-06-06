"""
Evaluation runner: generate predictions over a benchmark and score them.

The runner is decoupled from any specific model backend via the ``Generator``
protocol. ``ModelGenerator`` adapts the existing inference server for local
model directories, while tests can supply a lightweight callable to exercise
the scoring logic without downloading a model.
"""

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from nlm.eval.benchmark import CaseResult, EvalCase, EvalReport
from nlm.eval.metrics import DEFAULT_METRICS, METRIC_FUNCTIONS, keyword_recall

logger = logging.getLogger(__name__)


@runtime_checkable
class Generator(Protocol):
    """A callable that maps a prompt to a single generated completion."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class ModelGenerator:
    """
    Generator backed by a local model directory.

    Reuses :class:`nlm.inference.server.InferenceServer` for device selection,
    LoRA adapter loading, and generation so the harness scores models exactly
    as they will be served. Generation is deterministic (greedy) by default to
    keep evaluation reproducible.
    """

    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> None:
        # Imported lazily so metric-only callers need not import torch/flask.
        from nlm.inference.server import InferenceServer

        self.server = InferenceServer(model_dir)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        from nlm.inference.server import InferenceRequest

        # InferenceRequest.max_length is the total (prompt + generation) length,
        # so size it to the prompt plus the requested new tokens to avoid
        # truncating long prompts to empty generations. Capped at the schema max.
        if self.server.tokenizer is not None:
            prompt_len = len(self.server.tokenizer.encode(prompt))
        else:
            prompt_len = len(prompt) // 4
        max_len = min(prompt_len + self.max_new_tokens, 2048)

        request = InferenceRequest(
            prompt=prompt,
            max_length=max_len,
            do_sample=self.do_sample,
            temperature=self.temperature,
            num_return_sequences=1,
        )
        response = self.server.generate(request)
        return response.responses[0] if response.responses else ""


def _resolve_generate(generator: "Generator | Callable[[str], str]") -> Callable[[str], str]:
    """Return a ``prompt -> text`` callable from a Generator or plain callable."""
    if hasattr(generator, "generate"):
        return generator.generate  # type: ignore[attr-defined]
    if callable(generator):
        return generator
    raise TypeError("generator must implement .generate(prompt) or be callable")


def _score_case(
    case: EvalCase,
    prediction: str,
    metric_names: List[str],
) -> Dict[str, float]:
    """Compute the configured metric scores for one case."""
    scores: Dict[str, float] = {}

    # Reference-based metrics only apply when a gold reference is available.
    if case.reference:
        for name in metric_names:
            metric_fn = METRIC_FUNCTIONS.get(name)
            if metric_fn is None:
                logger.warning("Unknown metric '%s' - skipping", name)
                continue
            scores[name] = metric_fn(prediction, case.reference)

    # Keyword/behavioral check applies whenever keywords are specified.
    if case.keywords:
        scores["keyword_recall"] = keyword_recall(prediction, case.keywords)

    return scores


def evaluate(
    cases: List[EvalCase],
    generator: "Generator | Callable[[str], str]",
    metrics: Optional[List[str]] = None,
    threshold: float = 0.7,
    model_dir: str = "",
    benchmark: str = "",
) -> EvalReport:
    """
    Generate predictions for each case and score them.

    Args:
        cases: Evaluation cases to run.
        generator: Object with ``.generate(prompt)`` or a ``prompt -> str`` callable.
        metrics: Reference-based metric names to apply (defaults to DEFAULT_METRICS).
        threshold: Per-case pass threshold on the case's mean score, also used
            for the overall pass/fail decision.
        model_dir: Identifier recorded in the report (for provenance).
        benchmark: Benchmark path recorded in the report (for provenance).

    Returns:
        An :class:`EvalReport` with per-case and aggregate results.
    """
    if not cases:
        raise ValueError("No evaluation cases provided")

    metric_names = metrics or DEFAULT_METRICS
    generate_fn = _resolve_generate(generator)

    case_results: List[CaseResult] = []
    # Accumulators for weighted metric averages across cases.
    metric_totals: Dict[str, float] = {}
    metric_weights: Dict[str, float] = {}
    weighted_overall = 0.0
    total_weight = 0.0
    passed_count = 0

    for case in cases:
        prediction = generate_fn(case.prompt)
        scores = _score_case(case, prediction, metric_names)

        # Per-case overall is the mean of available metric scores. Cases with
        # no applicable metric (no reference and no keywords) score 0.0 and are
        # surfaced so the benchmark can be tightened.
        case_overall = sum(scores.values()) / len(scores) if scores else 0.0
        case_passed = bool(scores) and case_overall >= threshold

        case_results.append(
            CaseResult(
                id=case.id,
                prompt=case.prompt,
                prediction=prediction,
                reference=case.reference,
                scores=scores,
                overall=case_overall,
                passed=case_passed,
            )
        )

        for name, value in scores.items():
            metric_totals[name] = metric_totals.get(name, 0.0) + value * case.weight
            metric_weights[name] = metric_weights.get(name, 0.0) + case.weight

        weighted_overall += case_overall * case.weight
        total_weight += case.weight
        if case_passed:
            passed_count += 1

    metric_averages = {
        name: metric_totals[name] / metric_weights[name] for name in metric_totals
    }
    overall_score = weighted_overall / total_weight if total_weight else 0.0
    pass_rate = passed_count / len(cases)

    report = EvalReport(
        model_dir=model_dir,
        benchmark=benchmark,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_cases=len(cases),
        threshold=threshold,
        metric_averages=metric_averages,
        overall_score=overall_score,
        pass_rate=pass_rate,
        passed=overall_score >= threshold,
        case_results=case_results,
    )

    logger.info(
        "Evaluation complete: overall=%.3f pass_rate=%.1f%% (%d/%d cases)",
        overall_score,
        pass_rate * 100,
        passed_count,
        len(cases),
    )
    return report
