"""
Benchmark loading and result data models for the evaluation harness.

A benchmark is a JSONL file where each line describes one evaluation case.
Supported fields (all but ``prompt`` optional):

    prompt              -- the input given to the agent (required)
    reference|completion|expected -- gold answer for reference-based metrics
    keywords|expected_keywords    -- list of strings that should appear
    id                  -- stable case identifier (auto-assigned if absent)
    weight              -- relative weight for aggregate scoring (default 1.0)
    metadata            -- arbitrary dict carried through to the report
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Field aliases accepted from benchmark files, in priority order.
_REFERENCE_KEYS = ("reference", "completion", "expected", "answer")
_KEYWORD_KEYS = ("keywords", "expected_keywords", "must_contain")


class EvalCase(BaseModel):
    """A single evaluation case loaded from a benchmark file."""

    id: str = Field(..., description="Stable case identifier")
    prompt: str = Field(..., min_length=1, description="Input prompt for the agent")
    reference: Optional[str] = Field(
        default=None, description="Gold reference answer for reference-based metrics"
    )
    keywords: List[str] = Field(
        default_factory=list, description="Keywords/phrases expected in the output"
    )
    weight: float = Field(default=1.0, gt=0.0, description="Aggregate scoring weight")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary case metadata")


class CaseResult(BaseModel):
    """Scored result for a single evaluation case."""

    id: str
    prompt: str
    prediction: str
    reference: Optional[str] = None
    scores: Dict[str, float] = Field(default_factory=dict)
    overall: float = 0.0
    passed: bool = False


class EvalReport(BaseModel):
    """Aggregate evaluation report across all cases."""

    model_dir: str
    benchmark: str
    timestamp: str
    num_cases: int
    threshold: float
    metric_averages: Dict[str, float] = Field(default_factory=dict)
    overall_score: float = 0.0
    pass_rate: float = 0.0
    passed: bool = False
    fidelity: Dict[str, float] = Field(default_factory=dict)
    case_results: List[CaseResult] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for JSON serialization."""
        return self.model_dump()


def _first_present(record: Dict[str, Any], keys: tuple) -> Optional[Any]:
    """Return the first non-empty value among ``keys`` in ``record``."""
    for key in keys:
        if key in record and record[key] not in (None, "", []):
            return record[key]
    return None


def load_benchmark(path: str) -> List[EvalCase]:
    """
    Load evaluation cases from a JSONL benchmark file.

    Args:
        path: Path to a JSONL benchmark file.

    Returns:
        List of validated EvalCase objects.

    Raises:
        FileNotFoundError: If the benchmark file does not exist.
        ValueError: If the file contains no valid cases.
    """
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    cases: List[EvalCase] = []
    skipped = 0

    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping line %d: invalid JSON - %s", line_num, e)
                skipped += 1
                continue

            prompt = (record.get("prompt") or "").strip()
            if not prompt:
                logger.warning("Skipping line %d: missing prompt", line_num)
                skipped += 1
                continue

            reference = _first_present(record, _REFERENCE_KEYS)
            keywords = _first_present(record, _KEYWORD_KEYS) or []
            if isinstance(keywords, str):
                keywords = [keywords]

            case_id = str(record.get("id") or f"case-{line_num}")

            try:
                cases.append(
                    EvalCase(
                        id=case_id,
                        prompt=prompt,
                        reference=reference.strip() if isinstance(reference, str) else reference,
                        keywords=list(keywords),
                        weight=float(record.get("weight", 1.0)),
                        metadata=record.get("metadata", {}) or {},
                    )
                )
            except Exception as e:
                # Malformed records (e.g. non-positive weight) are skipped, not fatal.
                logger.warning("Skipping line %d: validation failed - %s", line_num, e)
                skipped += 1
                continue

    if skipped:
        logger.warning("Skipped %d invalid lines while loading benchmark", skipped)

    if not cases:
        raise ValueError(f"Benchmark contains no valid cases: {path}")

    logger.info("Loaded %d evaluation cases from %s", len(cases), path)
    return cases
