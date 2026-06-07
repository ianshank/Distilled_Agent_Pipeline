"""Unit tests for evaluation report serialization (JSON + HTML)."""

import json

from nlm.eval.benchmark import CaseResult, EvalReport
from nlm.eval.report import render_html_report, write_html_report, write_json_report


def _sample_report(**overrides) -> EvalReport:
    report = EvalReport(
        model_dir="m",
        benchmark="b",
        timestamp="2026-01-01T00:00:00Z",
        num_cases=1,
        threshold=0.5,
        metric_averages={"exact_match": 1.0},
        overall_score=1.0,
        pass_rate=1.0,
        passed=True,
        case_results=[
            CaseResult(
                id="c1",
                prompt="p",
                prediction="x",
                reference="x",
                scores={"exact_match": 1.0},
                overall=1.0,
                passed=True,
            )
        ],
    )
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


def test_write_json_report(temp_dir):
    path = temp_dir / "report.json"
    write_json_report(_sample_report(), str(path))
    data = json.loads(path.read_text())
    assert data["model_dir"] == "m"
    assert data["case_results"][0]["id"] == "c1"


def test_write_html_report(temp_dir):
    path = temp_dir / "report.html"
    write_html_report(_sample_report(), str(path))
    html = path.read_text()
    assert "Agent Evaluation Report" in html
    assert "c1" in html


def test_html_includes_fidelity_rows():
    report = _sample_report(fidelity={"top1_agreement": 0.9, "kl_divergence": 0.1})
    html = render_html_report(report)
    assert "fidelity/top1_agreement" in html


def test_html_marks_failed_cases():
    report = _sample_report(passed=False)
    report.case_results[0].passed = False
    html = render_html_report(report)
    assert "FAIL" in html
