"""
Command-line interface for the agent evaluation harness.

Examples:
    # Score a distilled SWE agent against a held-out benchmark
    python -m nlm.eval.cli \\
        --model-dir outputs/swe_agent/final \\
        --benchmark benchmarks/swe_agent_eval.jsonl \\
        --output-dir eval_outputs/swe_agent \\
        --threshold 0.6

The process exits non-zero when the overall score is below ``--threshold``,
so it can gate a CI pipeline. A GitHub Actions output file ($GITHUB_OUTPUT),
if present, receives ``overall_score``, ``pass_rate`` and ``passed``.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from nlm.eval.benchmark import EvalReport, load_benchmark
from nlm.eval.metrics import DEFAULT_METRICS
from nlm.eval.report import write_html_report, write_json_report
from nlm.eval.runner import ModelGenerator, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NLM distilled-agent evaluation harness")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--benchmark", required=True, help="Path to JSONL benchmark file")
    parser.add_argument(
        "--output-dir", default="eval_outputs", help="Directory for JSON/HTML reports"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Per-case and overall pass threshold on mean score",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated reference metrics (exact_match,token_f1,rouge_l)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max generation length"
    )
    parser.add_argument(
        "--do-sample", action="store_true", help="Enable sampling (default greedy/deterministic)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (if --do-sample)"
    )
    parser.add_argument(
        "--teacher-model-dir",
        default=None,
        help="Optional teacher model dir to compute teacher-student fidelity",
    )
    return parser


def _emit_github_output(report: EvalReport) -> None:
    """Write key metrics to the GitHub Actions output file when running in CI."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    try:
        with open(github_output, "a") as f:
            f.write(f"overall_score={report.overall_score:.4f}\n")
            f.write(f"pass_rate={report.pass_rate:.4f}\n")
            f.write(f"passed={str(report.passed).lower()}\n")
    except OSError as e:  # pragma: no cover - CI I/O edge case
        logger.warning("Could not write GITHUB_OUTPUT: %s", e)


def _print_summary(report: EvalReport) -> None:
    print("\n" + "=" * 60)
    print(f"Evaluation summary for {report.model_dir}")
    print(f"  Benchmark : {report.benchmark}")
    print(f"  Cases     : {report.num_cases}")
    print(f"  Threshold : {report.threshold:.2f}")
    for name, value in sorted(report.metric_averages.items()):
        print(f"  {name:<16}: {value:.3f}")
    for name, value in sorted(report.fidelity.items()):
        print(f"  fidelity/{name:<8}: {value:.3f}")
    print(f"  Overall   : {report.overall_score:.3f}")
    print(f"  Pass rate : {report.pass_rate * 100:.1f}%")
    print(f"  Result    : {'PASS' if report.passed else 'FAIL'}")
    print("=" * 60 + "\n")


def _maybe_add_fidelity(
    report: EvalReport, args: argparse.Namespace, generator: ModelGenerator
) -> None:
    """Compute teacher-student fidelity when a teacher dir is supplied."""
    if not args.teacher_model_dir:
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from nlm.eval.fidelity import evaluate_fidelity

        # Reuse the already-loaded student model rather than loading it a second
        # time, which would double memory use and risk OOM on large models.
        student = generator.server
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_dir)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_dir,
            torch_dtype=torch.float16 if student.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        teacher = teacher.to(student.device)

        prompts = [c.prompt for c in load_benchmark(args.benchmark)]
        report.fidelity = evaluate_fidelity(
            student.model, teacher, tokenizer, prompts, max_length=args.max_new_tokens
        )
    except Exception as e:  # pragma: no cover - heavy optional path
        logger.warning("Skipping fidelity computation: %s", e)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cases = load_benchmark(args.benchmark)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    logger.info("Loading model from %s", args.model_dir)
    generator = ModelGenerator(
        args.model_dir,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )

    report = evaluate(
        cases,
        generator,
        metrics=metric_names,
        threshold=args.threshold,
        model_dir=args.model_dir,
        benchmark=args.benchmark,
    )

    _maybe_add_fidelity(report, args, generator)

    out_dir = Path(args.output_dir)
    write_json_report(report, str(out_dir / "report.json"))
    write_html_report(report, str(out_dir / "report.html"))

    _print_summary(report)
    _emit_github_output(report)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
