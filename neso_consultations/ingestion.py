from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from neso_consultations.models import ColumnSpec, ConsultationData


_WHITESPACE_RE = re.compile(r"\s+")


def load_consultation_csv(path: Path) -> ConsultationData:
    """Load the consultation CSV into a normalized in-memory structure.

    Input:
        path: Local path to the source CSV file.

    Output:
        `ConsultationData` containing normalized column metadata and row values.

    Notes:
        Duplicate headers are handled by `_build_columns`, and short rows are
        padded to preserve index-safe access.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        raw_headers = next(reader)

        columns = _build_columns(raw_headers)
        rows: list[dict[str, str]] = []

        for row in reader:
            padded = row + [""] * (len(columns) - len(row))
            rows.append({col.unique_name: (padded[col.index] or "").strip() for col in columns})

    return ConsultationData(columns=columns, rows=rows)


def _build_columns(raw_headers: list[str]) -> list[ColumnSpec]:
    """Create unique column specs from raw headers.

    Input:
        raw_headers: Header row exactly as parsed from CSV.

    Output:
        List of `ColumnSpec` values where `unique_name` is deduplicated
        with `__N` suffixes for repeated header text.
    """
    name_counts: dict[str, int] = defaultdict(int)
    columns: list[ColumnSpec] = []

    for idx, raw_name in enumerate(raw_headers):
        normalized = _normalize_header(raw_name)
        name_counts[normalized] += 1
        duplicate_count = name_counts[normalized]

        unique_name = normalized if duplicate_count == 1 else f"{normalized}__{duplicate_count}"
        columns.append(ColumnSpec(unique_name=unique_name, raw_name=normalized, index=idx))

    return columns


def _normalize_header(header: str) -> str:
    """Normalize header text by removing invisible chars and extra spaces."""
    clean = header.replace("\ufeff", "")
    clean = clean.replace("\u200b", "")
    clean = _WHITESPACE_RE.sub(" ", clean).strip()
    return clean
