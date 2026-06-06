"""
Training-dataset validation.

A real validator (replacing the previous s3://-string stub) for JSONL training
data. Intentionally depends only on the standard library and pydantic so it can
run in a lightweight CI tier without torch/transformers, and gate training/CI.

Accepted record schemas mirror :mod:`nlm.data.dataset_loader`:
    * prompt/completion:  {"prompt": "...", "completion": "..."}
    * text:               {"text": "..."}

Importable API (``validate_dataset``) and a thin CLI (``python -m
nlm.data.validation`` / ``nlm-validate-data``).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default minimum sample count. Deliberately permissive (1) so it never silently
# rejects small curated datasets; callers/CLI override per dataset. No hardcoded
# project-specific threshold lives here.
DEFAULT_MIN_SAMPLES = 1


class DatasetValidationReport(BaseModel):
    """Structured result of validating a single dataset file."""

    path: str
    total_lines: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    empty_lines: int = 0
    duplicate_prompts: int = 0
    detected_schema: Optional[str] = None  # prompt_completion | text | mixed | None
    min_samples: int = DEFAULT_MIN_SAMPLES
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict:
        return self.model_dump()


def _classify_record(record: dict) -> Optional[str]:
    """Return the schema of a record, or None if it satisfies neither schema."""
    if not isinstance(record, dict):
        return None
    prompt = str(record.get("prompt", "")).strip()
    completion = str(record.get("completion", "")).strip()
    text = str(record.get("text", "")).strip()

    if prompt and completion:
        return "prompt_completion"
    if text:
        return "text"
    return None


def validate_dataset(
    path: str,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    allow_mixed_schema: bool = False,
    strict: bool = False,
) -> DatasetValidationReport:
    """
    Validate a JSONL training dataset.

    Checks performed:
        * file exists and is readable
        * each non-empty line is valid JSON
        * each record matches a supported schema with non-empty content
        * total valid records >= ``min_samples``
        * schema consistency across records (mixed schema is a warning, or an
          error when ``strict`` or not ``allow_mixed_schema``)
        * duplicate prompts/texts (warning)

    Args:
        path: Path to the JSONL file.
        min_samples: Minimum number of valid records required to pass.
        allow_mixed_schema: Permit a mix of prompt/completion and text records.
        strict: Treat warnings as failures.

    Returns:
        A :class:`DatasetValidationReport`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    report = DatasetValidationReport(path=str(file_path), min_samples=min_samples)
    schemas_seen = set()
    seen_keys = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            report.total_lines += 1

            if not line:
                report.empty_lines += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                report.invalid_records += 1
                report.errors.append(f"line {line_num}: invalid JSON - {e}")
                continue

            schema = _classify_record(record)
            if schema is None:
                report.invalid_records += 1
                report.errors.append(
                    f"line {line_num}: missing/empty required fields "
                    f"(need non-empty prompt+completion or text)"
                )
                continue

            report.valid_records += 1
            schemas_seen.add(schema)

            # Duplicate detection on the natural key for the schema.
            key = record.get("prompt") if schema == "prompt_completion" else record.get("text")
            if key in seen_keys:
                report.duplicate_prompts += 1
            else:
                seen_keys.add(key)

    # Determine detected schema label.
    if len(schemas_seen) == 1:
        report.detected_schema = next(iter(schemas_seen))
    elif len(schemas_seen) > 1:
        report.detected_schema = "mixed"

    # Sample-count check.
    if report.valid_records < min_samples:
        report.errors.append(
            f"too few valid records: {report.valid_records} < min_samples={min_samples}"
        )

    # Schema consistency.
    if report.detected_schema == "mixed" and not allow_mixed_schema:
        report.errors.append(
            "mixed record schemas detected (prompt/completion and text); "
            "pass allow_mixed_schema=True to permit"
        )

    # Non-fatal observations.
    if report.duplicate_prompts:
        report.warnings.append(
            f"{report.duplicate_prompts} duplicate prompt(s)/text(s) detected"
        )
    if report.invalid_records:
        report.warnings.append(f"{report.invalid_records} invalid record(s) skipped")

    has_blocking_warning = strict and bool(report.warnings)
    report.passed = not report.errors and not has_blocking_warning

    logger.info(
        "Validated %s: %d valid / %d invalid records, schema=%s, passed=%s",
        report.path,
        report.valid_records,
        report.invalid_records,
        report.detected_schema,
        report.passed,
    )
    return report


def _emit_github_output(reports: List[DatasetValidationReport]) -> None:
    """Write summary metrics to $GITHUB_OUTPUT when running under GitHub Actions."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    total_valid = sum(r.valid_records for r in reports)
    all_passed = all(r.passed for r in reports)
    try:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"total_valid_records={total_valid}\n")
            f.write(f"all_passed={str(all_passed).lower()}\n")
    except OSError as e:  # pragma: no cover - CI I/O edge case
        logger.warning("Could not write GITHUB_OUTPUT: %s", e)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate JSONL training datasets")
    parser.add_argument("--path", required=True, nargs="+", help="JSONL file path(s)")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum valid records required per file",
    )
    parser.add_argument(
        "--allow-mixed-schema",
        action="store_true",
        help="Permit a mix of prompt/completion and text records",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as failures"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON reports")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = build_parser().parse_args(argv)

    reports: List[DatasetValidationReport] = []
    for path in args.path:
        try:
            report = validate_dataset(
                path,
                min_samples=args.min_samples,
                allow_mixed_schema=args.allow_mixed_schema,
                strict=args.strict,
            )
        except FileNotFoundError as e:
            logger.error("%s", e)
            reports.append(
                DatasetValidationReport(path=path, errors=[str(e)], passed=False)
            )
            continue

        reports.append(report)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            status = "PASS" if report.passed else "FAIL"
            print(
                f"[{status}] {report.path}: {report.valid_records} valid "
                f"({report.detected_schema}), {len(report.errors)} error(s), "
                f"{len(report.warnings)} warning(s)"
            )
            for err in report.errors:
                print(f"    error: {err}")
            for warn in report.warnings:
                print(f"    warn:  {warn}")

    _emit_github_output(reports)
    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
