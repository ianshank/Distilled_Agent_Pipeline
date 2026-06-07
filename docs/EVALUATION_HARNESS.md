# Distilled Agent Evaluation Harness

A reproducible way to measure whether a distilled (student) agent actually
behaves the way you want, and how faithfully it tracks its teacher. The harness
lives in `NLM/nlm/eval/` and is exposed as both a Python API and a CLI
(`python -m nlm.eval.cli`).

## Why

Training computes a distillation loss, but loss alone does not tell you whether
the resulting agent answers prompts correctly. This harness scores a trained
model against a held-out **benchmark** using multiple complementary signals and
produces machine- and human-readable reports plus a pass/fail gate suitable for
CI.

## Benchmark format

A benchmark is a JSONL file, one evaluation case per line:

```json
{"id": "swe-fastapi-ping", "prompt": "Write a FastAPI /ping endpoint.", "reference": "from fastapi import FastAPI...", "keywords": ["FastAPI", "@app.get", "/ping"], "weight": 1.0, "metadata": {"category": "api"}}
```

| Field | Required | Purpose |
|-------|----------|---------|
| `prompt` | yes | Input given to the agent |
| `reference` / `completion` / `expected` | no | Gold answer for reference-based metrics |
| `keywords` / `expected_keywords` | no | Strings/phrases that must appear in the output |
| `id` | no | Stable identifier (auto-assigned if absent) |
| `weight` | no | Relative weight in aggregate scoring (default `1.0`) |
| `metadata` | no | Arbitrary dict carried through to the report |

A starter benchmark ships at `NLM/benchmarks/swe_agent_eval.jsonl`.

## Metrics

**Generation quality** (require a `reference`):

- `exact_match` — normalized string equality (0/1).
- `token_f1` — SQuAD-style token overlap F1.
- `rouge_l` — longest-common-subsequence F-measure.

**Behavioral** (require `keywords`):

- `keyword_recall` — fraction of expected keywords/phrases present. Matching is
  case-insensitive on the raw text so symbols like `q_proj` or `s3://` survive.

**Teacher-student fidelity** (optional, `--teacher-model-dir`):

- `top1_agreement` — fraction of positions where student and teacher pick the
  same next token (higher is better).
- `kl_divergence` — mean `KL(teacher || student)` over the prompt (lower is
  better). This is the distinctive distillation signal: it measures how closely
  the student reproduces the teacher's distribution, mirroring the temperature
  scaling used in the training loss.

Each case's **overall** score is the mean of its applicable metric scores; a
case passes when that mean meets `--threshold`. The report's overall score is
the weighted mean across cases, and `pass_rate` is the fraction of passing
cases.

## CLI usage

> **Dependencies:** the CLI loads the model via the inference server
> (torch/transformers/flask), so it requires the `[ml]` extra:
> `pip install -e ".[ml]"`. The pure-Python scoring metrics and benchmark
> loading work without it.

```bash
cd NLM
python -m nlm.eval.cli \
  --model-dir outputs/swe_agent/final \
  --benchmark benchmarks/swe_agent_eval.jsonl \
  --output-dir eval_outputs/swe_agent \
  --threshold 0.6

# With teacher-student fidelity
python -m nlm.eval.cli \
  --model-dir outputs/swe_agent/final \
  --teacher-model-dir ibm-granite/granite-3.0-8b-instruct \
  --benchmark benchmarks/swe_agent_eval.jsonl
```

Outputs:

- `eval_outputs/.../report.json` — full machine-readable report.
- `eval_outputs/.../report.html` — human-readable report with per-case detail.
- Console summary table.
- When run under GitHub Actions, `overall_score`, `pass_rate`, and `passed` are
  written to `$GITHUB_OUTPUT`.

The process **exits non-zero when the overall score is below the threshold**, so
it can gate a pipeline directly.

## Python API

```python
from nlm.eval import load_benchmark, evaluate, write_json_report

cases = load_benchmark("benchmarks/swe_agent_eval.jsonl")

# Any object with .generate(prompt) -> str, or a plain callable, works.
report = evaluate(cases, my_generator, threshold=0.7)
write_json_report(report, "report.json")
print(report.overall_score, report.pass_rate, report.passed)
```

`ModelGenerator` adapts a local model directory by reusing the inference
server's loading/generation logic, so the harness scores a model exactly as it
will be served. Generation is deterministic (greedy) by default for
reproducibility.

## Testing

The scoring logic is pure Python and fully unit tested without a model or
network:

```bash
cd NLM
pytest tests/test_eval_metrics.py tests/test_eval_benchmark.py tests/test_eval_runner.py -v
# Fidelity tests require torch:
pytest tests/test_eval_fidelity.py -v
```

## Extending

- **Add a metric:** implement `fn(prediction, reference) -> float` in
  `nlm/eval/metrics.py` and register it in `METRIC_FUNCTIONS`.
- **New backend:** pass any object implementing `.generate(prompt) -> str` to
  `evaluate` (e.g. an API client or an LLM-as-judge wrapper).
- **Per-role benchmarks:** add `NLM/benchmarks/<role>_eval.jsonl` and point the
  CLI at it.
