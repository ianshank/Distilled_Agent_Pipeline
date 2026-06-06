"""Unit tests for benchmark loading and result models."""

import json

import pytest

from nlm.eval.benchmark import EvalReport, load_benchmark


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestLoadBenchmark:
    def test_loads_prompt_and_reference(self, temp_dir):
        path = temp_dir / "bench.jsonl"
        _write_jsonl(path, [{"prompt": "hi", "completion": "hello there"}])

        cases = load_benchmark(str(path))
        assert len(cases) == 1
        assert cases[0].prompt == "hi"
        assert cases[0].reference == "hello there"

    def test_reference_alias_priority(self, temp_dir):
        path = temp_dir / "bench.jsonl"
        _write_jsonl(path, [{"prompt": "p", "reference": "R", "completion": "C"}])
        cases = load_benchmark(str(path))
        # "reference" wins over "completion".
        assert cases[0].reference == "R"

    def test_keywords_parsed(self, temp_dir):
        path = temp_dir / "bench.jsonl"
        _write_jsonl(path, [{"prompt": "p", "expected_keywords": ["a", "b"]}])
        cases = load_benchmark(str(path))
        assert cases[0].keywords == ["a", "b"]

    def test_string_keyword_coerced_to_list(self, temp_dir):
        path = temp_dir / "bench.jsonl"
        _write_jsonl(path, [{"prompt": "p", "keywords": "solo"}])
        cases = load_benchmark(str(path))
        assert cases[0].keywords == ["solo"]

    def test_auto_id_assigned(self, temp_dir):
        path = temp_dir / "bench.jsonl"
        _write_jsonl(path, [{"prompt": "p"}])
        cases = load_benchmark(str(path))
        assert cases[0].id == "case-1"

    def test_skips_invalid_and_empty_lines(self, temp_dir):
        path = temp_dir / "bench.jsonl"
        with open(path, "w") as f:
            f.write('{"prompt": "good"}\n')
            f.write("\n")
            f.write("not json\n")
            f.write('{"no_prompt": "x"}\n')
        cases = load_benchmark(str(path))
        assert len(cases) == 1

    def test_missing_file_raises(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            load_benchmark(str(temp_dir / "nope.jsonl"))

    def test_empty_benchmark_raises(self, temp_dir):
        path = temp_dir / "empty.jsonl"
        path.write_text("\n\n")
        with pytest.raises(ValueError):
            load_benchmark(str(path))


class TestEvalReport:
    def test_to_dict_roundtrip(self):
        report = EvalReport(
            model_dir="m",
            benchmark="b",
            timestamp="2026-01-01T00:00:00Z",
            num_cases=0,
            threshold=0.7,
        )
        d = report.to_dict()
        assert d["model_dir"] == "m"
        assert d["threshold"] == 0.7
        assert "case_results" in d
