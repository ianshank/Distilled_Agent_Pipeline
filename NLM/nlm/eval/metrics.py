"""
Reference-based scoring metrics for the agent evaluation harness.

All functions in this module are pure (no model or framework dependency) and
return a float score in the range [0.0, 1.0] where higher is better. This keeps
them fast to unit test and safe to import in environments without torch.
"""

import string
from typing import Callable, Dict, List, Sequence

# Articles stripped during normalization (SQuAD-style) so trivial wording
# differences do not penalize otherwise-correct answers.
_ARTICLES = {"a", "an", "the"}


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Lowercases, removes punctuation and articles, and collapses whitespace.

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    text = text.lower()
    # Drop punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace and remove articles
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def _tokenize(text: str) -> List[str]:
    """Tokenize normalized text into word tokens."""
    return normalize_text(text).split()


def exact_match(prediction: str, reference: str) -> float:
    """
    Exact match after normalization.

    Args:
        prediction: Model-generated text.
        reference: Gold reference text.

    Returns:
        1.0 if normalized strings are identical, else 0.0.
    """
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 score (SQuAD-style).

    Computes the harmonic mean of token precision and recall over the
    multiset of normalized tokens.

    Args:
        prediction: Model-generated text.
        reference: Gold reference text.

    Returns:
        F1 score in [0.0, 1.0].
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Count overlapping tokens as a multiset intersection.
    common = 0
    ref_counts: Dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    for tok in pred_tokens:
        if ref_counts.get(tok, 0) > 0:
            common += 1
            ref_counts[tok] -= 1

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of two token sequences."""
    if not a or not b:
        return 0

    # Space-optimized DP using two rows.
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L F-measure based on longest common subsequence.

    Args:
        prediction: Model-generated text.
        reference: Gold reference text.

    Returns:
        ROUGE-L F1 score in [0.0, 1.0].
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(prediction: str, keywords: List[str]) -> float:
    """
    Fraction of expected keywords/phrases present in the prediction.

    Useful for behavioral checks where an agent answer must mention specific
    concepts, APIs, or tokens regardless of exact phrasing. Matching is
    case-insensitive on the raw (un-normalized) text so that symbols such as
    ``q_proj`` or ``s3://`` are preserved.

    Args:
        prediction: Model-generated text.
        keywords: Keywords or phrases that should appear in the prediction.

    Returns:
        Fraction of keywords found, in [0.0, 1.0]. Returns 1.0 when the
        keyword list is empty (nothing required).
    """
    if not keywords:
        return 1.0

    haystack = prediction.lower()
    found = sum(1 for kw in keywords if kw.lower() in haystack)
    return found / len(keywords)


# Registry of reference-based metrics keyed by name. Each callable takes
# (prediction, reference) and returns a float in [0.0, 1.0].
METRIC_FUNCTIONS: Dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
}

# Metrics applied by default when the caller does not specify a subset.
DEFAULT_METRICS: List[str] = ["exact_match", "token_f1", "rouge_l"]
