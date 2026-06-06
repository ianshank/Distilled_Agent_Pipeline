"""Unit tests for dataset validation (pure-Python, no torch)."""

import json

import pytest

from nlm.data.validation import DEFAULT_MIN_SAMPLES, validate_dataset


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write((rec if isinstance(rec, str) else json.dumps(rec)) + "\n")


class TestValidateDataset:
    def test_valid_prompt_completion(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(
            path,
            [
                {"prompt": "a", "completion": "x"},
                {"prompt": "b", "completion": "y"},
            ],
        )
        report = validate_dataset(str(path))
        assert report.passed is True
        assert report.valid_records == 2
        assert report.invalid_records == 0
        assert report.detected_schema == "prompt_completion"

    def test_valid_text_schema(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"text": "hello"}, {"text": "world"}])
        report = validate_dataset(str(path))
        assert report.passed is True
        assert report.detected_schema == "text"

    def test_invalid_json_line_is_error(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"prompt": "a", "completion": "x"}, "not json{"])
        report = validate_dataset(str(path))
        assert report.invalid_records == 1
        assert report.passed is False
        assert any("invalid JSON" in e for e in report.errors)

    def test_missing_required_fields_is_error(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"prompt": "a"}, {"foo": "bar"}])
        report = validate_dataset(str(path))
        assert report.invalid_records == 2
        assert report.passed is False

    def test_empty_values_rejected(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"prompt": "  ", "completion": ""}])
        report = validate_dataset(str(path))
        assert report.valid_records == 0
        assert report.passed is False

    def test_empty_lines_ignored(self, temp_dir):
        path = temp_dir / "d.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "a"}) + "\n")
            f.write("\n")
            f.write("   \n")
        report = validate_dataset(str(path))
        assert report.empty_lines == 2
        assert report.valid_records == 1

    def test_min_samples_enforced(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"text": "a"}])
        report = validate_dataset(str(path), min_samples=5)
        assert report.passed is False
        assert any("too few valid records" in e for e in report.errors)

    def test_duplicate_prompts_warned_not_fatal(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(
            path,
            [
                {"prompt": "dup", "completion": "x"},
                {"prompt": "dup", "completion": "y"},
            ],
        )
        report = validate_dataset(str(path))
        assert report.duplicate_prompts == 1
        assert report.passed is True  # duplicates are a warning by default
        assert any("duplicate" in w for w in report.warnings)

    def test_mixed_schema_is_error_by_default(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"prompt": "a", "completion": "x"}, {"text": "b"}])
        report = validate_dataset(str(path))
        assert report.detected_schema == "mixed"
        assert report.passed is False

    def test_mixed_schema_allowed(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"prompt": "a", "completion": "x"}, {"text": "b"}])
        report = validate_dataset(str(path), allow_mixed_schema=True)
        assert report.passed is True

    def test_strict_treats_warnings_as_failure(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(
            path,
            [
                {"prompt": "dup", "completion": "x"},
                {"prompt": "dup", "completion": "y"},
            ],
        )
        report = validate_dataset(str(path), strict=True)
        assert report.passed is False

    def test_unhashable_key_does_not_crash(self, temp_dir):
        # A list/dict prompt is coerced to a string for duplicate detection
        # rather than raising TypeError on an unhashable key.
        path = temp_dir / "d.jsonl"
        _write_jsonl(
            path,
            [
                {"prompt": ["a", "b"], "completion": "x"},
                {"prompt": ["a", "b"], "completion": "y"},
            ],
        )
        report = validate_dataset(str(path))
        assert report.valid_records == 2
        assert report.duplicate_prompts == 1

    def test_missing_file_raises(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            validate_dataset(str(temp_dir / "nope.jsonl"))

    def test_default_min_samples_constant(self):
        assert DEFAULT_MIN_SAMPLES == 1

    def test_to_dict_serializable(self, temp_dir):
        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"text": "a"}])
        report = validate_dataset(str(path))
        d = report.to_dict()
        assert d["path"].endswith("d.jsonl")
        assert d["valid_records"] == 1


class TestCLI:
    def test_cli_pass(self, temp_dir, capsys):
        from nlm.data.validation import main

        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"text": "a"}, {"text": "b"}])
        rc = main(["--path", str(path)])
        assert rc == 0

    def test_cli_fail_min_samples(self, temp_dir):
        from nlm.data.validation import main

        path = temp_dir / "d.jsonl"
        _write_jsonl(path, [{"text": "a"}])
        rc = main(["--path", str(path), "--min-samples", "10"])
        assert rc == 1

    def test_cli_missing_file_returns_one(self, temp_dir):
        from nlm.data.validation import main

        rc = main(["--path", str(temp_dir / "nope.jsonl")])
        assert rc == 1
