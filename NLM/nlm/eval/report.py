"""
Report serialization for the evaluation harness.

Produces a machine-readable JSON report and a human-readable HTML report from
an :class:`nlm.eval.benchmark.EvalReport`.
"""

import html
import json
import logging
from pathlib import Path

from nlm.eval.benchmark import EvalReport

logger = logging.getLogger(__name__)


def write_json_report(report: EvalReport, path: str) -> None:
    """Write the report as indented JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Wrote JSON report to %s", out_path)


def _metric_rows(report: EvalReport) -> str:
    rows = []
    for name, value in sorted(report.metric_averages.items()):
        rows.append(
            f'<div class="metric"><strong>{html.escape(name)}:</strong> {value:.3f}</div>'
        )
    for name, value in sorted(report.fidelity.items()):
        rows.append(
            f'<div class="metric"><strong>fidelity/{html.escape(name)}:</strong> {value:.3f}</div>'
        )
    return "\n".join(rows)


def _case_rows(report: EvalReport, max_cases: int = 200) -> str:
    rows = []
    for case in report.case_results[:max_cases]:
        status_class = "case-pass" if case.passed else "case-fail"
        status_emoji = "PASS" if case.passed else "FAIL"
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in case.scores.items()) or "n/a"
        rows.append(
            f"""
    <div class="case {status_class}">
        <h4>[{status_emoji}] {html.escape(case.id)} &mdash; overall {case.overall:.2f}</h4>
        <p class="scores">{html.escape(score_str)}</p>
        <details><summary>prompt</summary><pre>{html.escape(case.prompt)}</pre></details>
        <details><summary>prediction</summary><pre>{html.escape(case.prediction)}</pre></details>
    </div>"""
        )
    return "\n".join(rows)


def render_html_report(report: EvalReport) -> str:
    """Render the report as a standalone HTML string."""
    overall_class = "pass" if report.passed else "fail"
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Agent Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, Arial, sans-serif; margin: 40px; color: #222; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 8px; padding: 12px;
                   background: #eef3fb; border-radius: 6px; }}
        .summary.pass {{ border-left: 6px solid #2e7d32; padding-left: 12px; }}
        .summary.fail {{ border-left: 6px solid #c62828; padding-left: 12px; }}
        .case {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .case-pass {{ border-color: #2e7d32; }}
        .case-fail {{ border-color: #c62828; }}
        .scores {{ color: #555; font-family: monospace; }}
        pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 8px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Agent Evaluation Report</h1>
        <p><strong>Model:</strong> {html.escape(report.model_dir or "n/a")}</p>
        <p><strong>Benchmark:</strong> {html.escape(report.benchmark or "n/a")}</p>
        <p><strong>Generated:</strong> {html.escape(report.timestamp)}</p>
    </div>

    <h2>Summary</h2>
    <div class="summary {overall_class}">
        <p><strong>Result:</strong> {"PASSED" if report.passed else "FAILED"}
           (overall {report.overall_score:.3f} vs threshold {report.threshold:.2f})</p>
        <p><strong>Pass rate:</strong> {report.pass_rate * 100:.1f}% over {report.num_cases} cases</p>
    </div>

    <h3>Metric Averages</h3>
    {_metric_rows(report)}

    <h3>Case Results</h3>
    {_case_rows(report)}
</body>
</html>
"""


def write_html_report(report: EvalReport, path: str) -> None:
    """Write the report as an HTML file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_html_report(report))
    logger.info("Wrote HTML report to %s", out_path)
