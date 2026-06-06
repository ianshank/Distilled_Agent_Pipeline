"""Unit tests for evaluation metric functions."""

import pytest

from nlm.eval.metrics import (
    DEFAULT_METRICS,
    METRIC_FUNCTIONS,
    exact_match,
    keyword_recall,
    normalize_text,
    rouge_l,
    token_f1,
)


class TestNormalizeText:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_removes_articles(self):
        assert normalize_text("the quick a fox an apple") == "quick fox apple"

    def test_collapses_whitespace(self):
        assert normalize_text("  multiple   spaces\nhere ") == "multiple spaces here"

    def test_empty_input(self):
        assert normalize_text("") == ""


class TestExactMatch:
    def test_identical_after_normalization(self):
        assert exact_match("The Answer.", "answer") == 1.0

    def test_mismatch(self):
        assert exact_match("cat", "dog") == 0.0

    def test_returns_float(self):
        assert isinstance(exact_match("a", "a"), float)


class TestTokenF1:
    def test_perfect_overlap(self):
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("cat dog", "fish bird") == 0.0

    def test_partial_overlap(self):
        # pred: {quick, brown, fox}, ref: {quick, fox}
        score = token_f1("quick brown fox", "quick fox")
        # precision 2/3, recall 2/2 -> F1 = 2*(2/3)/(1+2/3) = 0.8
        assert score == pytest.approx(0.8)

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("", "something") == 0.0

    def test_bounded(self):
        assert 0.0 <= token_f1("a b c d", "a x y z") <= 1.0


class TestRougeL:
    def test_identical(self):
        assert rouge_l("a b c d", "a b c d") == pytest.approx(1.0)

    def test_subsequence_order_matters(self):
        # tokens: [one, two, three] vs [one, three, two]; LCS length 2
        # precision 2/3, recall 2/3 -> F1 = 2/3
        score = rouge_l("one two three", "one three two")
        assert score == pytest.approx(2 / 3)

    def test_no_overlap(self):
        assert rouge_l("cat dog fish", "lion tiger bear") == 0.0

    def test_empty(self):
        assert rouge_l("", "") == 1.0
        assert rouge_l("word", "") == 0.0


class TestLcsHelper:
    def test_lcs_empty_returns_zero(self):
        from nlm.eval.metrics import _lcs_length

        assert _lcs_length([], ["a"]) == 0
        assert _lcs_length(["a"], []) == 0


class TestKeywordRecall:
    def test_all_present(self):
        assert keyword_recall("import FastAPI from app", ["FastAPI", "import"]) == 1.0

    def test_half_present(self):
        assert keyword_recall("only jwt here", ["jwt", "oauth"]) == pytest.approx(0.5)

    def test_none_present(self):
        assert keyword_recall("nothing matches", ["zzz", "qqq"]) == 0.0

    def test_empty_keywords_returns_one(self):
        assert keyword_recall("anything", []) == 1.0

    def test_case_insensitive(self):
        assert keyword_recall("USERS table", ["users"]) == 1.0

    def test_preserves_symbols(self):
        # Normalization would strip these symbols; keyword match must not.
        assert keyword_recall("use s3:// and q_proj", ["s3://", "q_proj"]) == 1.0


class TestRegistry:
    def test_default_metrics_are_registered(self):
        for name in DEFAULT_METRICS:
            assert name in METRIC_FUNCTIONS

    def test_registry_callables_return_float(self):
        for fn in METRIC_FUNCTIONS.values():
            assert isinstance(fn("x", "x"), float)
