"""Unit tests for the evaluation runner using a fake generator (no model)."""

import pytest

from nlm.eval.benchmark import EvalCase
from nlm.eval.report import render_html_report, write_json_report
from nlm.eval.runner import evaluate


class EchoGenerator:
    """Generator that returns a canned mapping or echoes the prompt."""

    def __init__(self, responses=None):
        self.responses = responses or {}

    def generate(self, prompt: str) -> str:
        return self.responses.get(prompt, prompt)


def test_perfect_reference_match_passes():
    cases = [EvalCase(id="c1", prompt="p1", reference="the answer")]
    gen = EchoGenerator({"p1": "the answer"})

    report = evaluate(cases, gen, threshold=0.7)

    assert report.num_cases == 1
    assert report.case_results[0].passed is True
    assert report.overall_score == pytest.approx(1.0)
    assert report.pass_rate == pytest.approx(1.0)
    assert report.passed is True


def test_keyword_only_scoring():
    cases = [EvalCase(id="c1", prompt="p1", keywords=["alpha", "beta"])]
    gen = EchoGenerator({"p1": "contains alpha but not the other"})

    report = evaluate(cases, gen, threshold=0.7)

    # 1 of 2 keywords -> 0.5, below threshold.
    assert report.case_results[0].scores["keyword_recall"] == pytest.approx(0.5)
    assert report.case_results[0].passed is False


def test_callable_generator_accepted():
    cases = [EvalCase(id="c1", prompt="echo", reference="echo")]
    report = evaluate(cases, lambda p: p, threshold=0.5)
    assert report.passed is True


def test_case_with_no_metrics_scores_zero():
    cases = [EvalCase(id="c1", prompt="p1")]  # no reference, no keywords
    report = evaluate(cases, EchoGenerator(), threshold=0.1)
    assert report.case_results[0].overall == 0.0
    assert report.case_results[0].passed is False


def test_weighted_average_across_cases():
    cases = [
        EvalCase(id="c1", prompt="p1", reference="match", weight=3.0),
        EvalCase(id="c2", prompt="p2", reference="nomatch", weight=1.0),
    ]
    gen = EchoGenerator({"p1": "match", "p2": "totally different words"})
    report = evaluate(cases, gen, threshold=0.7)

    # c1 overall ~1.0 (weight 3), c2 low (weight 1) -> weighted overall high.
    assert report.overall_score > 0.7
    assert report.pass_rate == pytest.approx(0.5)


def test_unknown_metric_is_skipped():
    cases = [EvalCase(id="c1", prompt="p1", reference="x")]
    report = evaluate(cases, EchoGenerator({"p1": "x"}), metrics=["bogus", "exact_match"])
    assert "bogus" not in report.case_results[0].scores
    assert "exact_match" in report.case_results[0].scores


def test_empty_cases_raises():
    with pytest.raises(ValueError):
        evaluate([], EchoGenerator())


def test_resolve_generate_rejects_non_callable():
    from nlm.eval.runner import _resolve_generate

    with pytest.raises(TypeError):
        _resolve_generate(object())


def test_metric_averages_only_include_applicable():
    cases = [
        EvalCase(id="c1", prompt="p1", reference="r"),
        EvalCase(id="c2", prompt="p2", keywords=["k"]),
    ]
    gen = EchoGenerator({"p1": "r", "p2": "k"})
    report = evaluate(cases, gen)
    assert "exact_match" in report.metric_averages
    assert "keyword_recall" in report.metric_averages


def test_reports_serialize(temp_dir):
    cases = [EvalCase(id="c1", prompt="p1", reference="p1")]
    report = evaluate(cases, lambda p: p, model_dir="m", benchmark="b")

    json_path = temp_dir / "report.json"
    write_json_report(report, str(json_path))
    assert json_path.exists()

    html = render_html_report(report)
    assert "Agent Evaluation Report" in html
    assert "c1" in html
