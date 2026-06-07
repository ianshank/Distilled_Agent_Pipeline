"""Data loading, preprocessing, and validation for NLM distillation.

``load_distillation_dataset`` / ``convert_jsonl_format`` are imported lazily
(PEP 562) so that lightweight consumers — e.g. dataset validation — can import
``nlm.data`` without pulling in the heavy ``transformers``/``datasets`` stack.
"""

from typing import TYPE_CHECKING

from nlm.data.validation import DatasetValidationReport, validate_dataset

__all__ = [
    "load_distillation_dataset",
    "convert_jsonl_format",
    "validate_dataset",
    "DatasetValidationReport",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nlm.data.dataset_loader import convert_jsonl_format, load_distillation_dataset

_LAZY = {"load_distillation_dataset", "convert_jsonl_format"}


def __getattr__(name):
    """Lazily resolve heavy dataset-loader symbols on first access."""
    if name in _LAZY:
        from nlm.data import dataset_loader

        return getattr(dataset_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
