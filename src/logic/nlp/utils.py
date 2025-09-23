from __future__ import annotations

import re
import threading
import unicodedata
from pathlib import Path
from typing import FrozenSet, Iterable, List, Optional, Tuple

import pandas as pd


# Module-level caches for fast O(1) membership checks
_LOCATIONS_NORMALIZED: Optional[FrozenSet[str]] = None
_LOCATIONS_ORIGINAL: Optional[FrozenSet[str]] = None
_LOAD_LOCK = threading.Lock()


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_excel_path() -> Path:
    return _module_dir() / "location.xlsx"


def _normalize_whitespace(text: str) -> str:
    # Collapse all whitespace to single spaces and strip ends
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """Normalize text for robust matching.

    - Lowercase
    - Remove Vietnamese diacritics (and all combining marks)
    - Replace punctuation-like separators with single spaces
    - Collapse repeated whitespace
    """
    if text is None:
        return ""

    lowered = str(text).lower()
    # Replace common separators with spaces for consistency
    replaced = re.sub(r"[_\-.,/\\]+", " ", lowered)
    # Unicode NFD decomposition to split base chars and diacritics
    decomposed = unicodedata.normalize("NFD", replaced)
    # Remove combining marks (diacritics)
    without_diacritics = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    # Recompose to NFC and normalize spaces
    recomposed = unicodedata.normalize("NFC", without_diacritics)
    return _normalize_whitespace(recomposed)


def _extract_strings_from_dataframe(df: pd.DataFrame) -> List[str]:
    # Select only object/string-like columns
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(string_cols) == 0:
        return []
    series = df[string_cols].astype(str).stack(dropna=True)
    # Filter out empty/placeholder values
    values = [v for v in series.tolist() if v and v.lower() not in {"nan", "none", "null"}]
    return values


def _read_excel_locations(excel_path: Optional[Path] = None) -> List[str]:
    path = excel_path or _default_excel_path()
    if not path.exists():
        raise FileNotFoundError(f"Location Excel file not found at: {path}")

    # Read all sheets to be robust to future changes
    sheets = pd.read_excel(path, sheet_name=None)  # type: ignore[arg-type]
    values: List[str] = []
    for _, df in sheets.items():
        values.extend(_extract_strings_from_dataframe(df))
    # Deduplicate while preserving order
    seen = set()
    unique_values: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique_values.append(v)
    return unique_values


def _build_location_sets(raw_values: Iterable[str]) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    originals: List[str] = []
    normalized: List[str] = []

    for value in raw_values:
        if not value:
            continue
        value_str = str(value)
        norm = normalize_text(value_str)
        if not norm:
            continue
        originals.append(value_str)
        normalized.append(norm)

    return frozenset(originals), frozenset(normalized)


def _ensure_loaded(excel_path: Optional[Path] = None) -> None:
    global _LOCATIONS_ORIGINAL, _LOCATIONS_NORMALIZED
    if _LOCATIONS_NORMALIZED is not None:
        return
    with _LOAD_LOCK:
        if _LOCATIONS_NORMALIZED is not None:
            return
        raw_values = _read_excel_locations(excel_path)
        originals, normalized = _build_location_sets(raw_values)
        _LOCATIONS_ORIGINAL = originals
        _LOCATIONS_NORMALIZED = normalized


def is_known_location(text: str, *, excel_path: Optional[str] = None) -> bool:
    """Return True if the given string matches a known location.

    The match is accent-insensitive and case-insensitive. Whitespace and common
    separators are normalized.
    """
    path = Path(excel_path) if excel_path else None
    _ensure_loaded(path)
    if not text:
        return False
    assert _LOCATIONS_NORMALIZED is not None
    return normalize_text(text) in _LOCATIONS_NORMALIZED


def all_locations(*, normalized: bool = False, excel_path: Optional[str] = None) -> FrozenSet[str]:
    """Return all loaded locations.

    - If normalized=True, returns the normalized set used for matching.
    - Otherwise, returns original values from the Excel.
    """
    path = Path(excel_path) if excel_path else None
    _ensure_loaded(path)
    assert _LOCATIONS_ORIGINAL is not None and _LOCATIONS_NORMALIZED is not None
    return _LOCATIONS_NORMALIZED if normalized else _LOCATIONS_ORIGINAL


def count_locations(excel_path: Optional[str] = None) -> int:
    path = Path(excel_path) if excel_path else None
    _ensure_loaded(path)
    assert _LOCATIONS_NORMALIZED is not None
    return len(_LOCATIONS_NORMALIZED)


def reload_locations(excel_path: Optional[str] = None) -> None:
    """Force reload from Excel. Use if the file is updated at runtime."""
    path = Path(excel_path) if excel_path else None
    with _LOAD_LOCK:
        raw_values = _read_excel_locations(path)
        originals, normalized = _build_location_sets(raw_values)
        global _LOCATIONS_ORIGINAL, _LOCATIONS_NORMALIZED
        _LOCATIONS_ORIGINAL = originals
        _LOCATIONS_NORMALIZED = normalized


