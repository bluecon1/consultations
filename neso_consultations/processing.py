from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from neso_consultations.models import (
    ColumnSpec,
    ConsultationData,
    OrganisationCatalog,
    PreparedData,
    QuestionDefinition,
    ResponseItem,
)


SECTION_MARKERS = {"Strategic Investment Need", "Overall"}
SUPPLEMENT_PREFIXES = (
    "if ",
    "please provide",
    "if not",
    "if you",
)
CATEGORICAL_HINTS = {
    "strongly agree",
    "somewhat agree",
    "neither agree nor disagree",
    "somewhat disagree",
    "strongly disagree",
    "yes",
    "no",
    "maybe",
    "agree",
    "disagree",
    "neutral",
    "no comment",
}


@dataclass(frozen=True)
class QuestionSlice:
    question: QuestionDefinition
    items: list[ResponseItem]


def prepare_data(
    consultation_data: ConsultationData,
    *,
    excerpt_chars: int = 280,
    section_mapping_path: Path | None = None,
) -> PreparedData:
    """Transform raw CSV structures into normalized question and response objects.

    Inputs:
        consultation_data: Parsed CSV rows and deduplicated columns.
        excerpt_chars: Maximum per-response excerpt length for prompts/UI.
        section_mapping_path: Optional XLSX path mapping survey headers to
            section names. When provided and readable, this mapping is used as
            the section source of truth.

    Output:
        `PreparedData` containing question definitions and response items.
    """
    section_by_index = load_section_mapping(consultation_data.columns, section_mapping_path)
    questions = build_question_definitions(consultation_data.columns, section_by_index=section_by_index)
    items = build_response_items(consultation_data, questions, excerpt_chars=excerpt_chars)
    return PreparedData(consultation_data=consultation_data, questions=questions, response_items=items)


def build_question_definitions(
    columns: list[ColumnSpec],
    *,
    section_by_index: dict[int, str] | None = None,
) -> list[QuestionDefinition]:
    """Infer logical question blocks from the flat survey column list.

    Input:
        columns: Normalized `ColumnSpec` metadata from ingestion.
        section_by_index: Optional section mapping keyed by CSV column index.

    Output:
        Ordered `QuestionDefinition` list with section and supplemental mappings.

    Notes:
        Supplemental headers (reasoning text, yes/maybe/no free text) are
        attached to the most recent primary question column.
    """
    section_by_index = section_by_index or {}
    question_start_idx = _find_question_start_index(columns)
    question_columns = columns[question_start_idx:]

    questions: list[QuestionDefinition] = []
    current_question: QuestionDefinition | None = None
    current_section = "General"

    for column in question_columns:
        raw = _clean_text(column.raw_name)
        lowered = raw.lower()
        mapped_section = _clean_text(section_by_index.get(column.index, ""))

        if raw in SECTION_MARKERS:
            current_section = mapped_section or raw
            continue

        if _is_supplemental_header(lowered):
            if current_question is None:
                current_question = QuestionDefinition(
                    question_id=f"Q{len(questions) + 1:02d}",
                    question_text=raw,
                    section=current_section,
                    primary_column=column,
                    supplemental_columns=[],
                )
                questions.append(current_question)
                continue

            supplements = [*current_question.supplemental_columns, column]
            current_question = QuestionDefinition(
                question_id=current_question.question_id,
                question_text=current_question.question_text,
                section=current_question.section,
                primary_column=current_question.primary_column,
                supplemental_columns=supplements,
            )
            questions[-1] = current_question
            continue

        current_question = QuestionDefinition(
            question_id=f"Q{len(questions) + 1:02d}",
            question_text=_canonical_question_text(raw),
            section=mapped_section or current_section,
            primary_column=column,
            supplemental_columns=[],
        )
        if mapped_section:
            current_section = mapped_section
        questions.append(current_question)

    return questions


def build_response_items(
    consultation_data: ConsultationData,
    questions: list[QuestionDefinition],
    *,
    excerpt_chars: int,
) -> list[ResponseItem]:
    """Create per-question response records for downstream summarisation.

    Inputs:
        consultation_data: Source rows/columns.
        questions: Canonical question definitions.
        excerpt_chars: Max characters kept in `ResponseItem.excerpt`.

    Output:
        Flat list of `ResponseItem`, one per answered question per submission.

    Notes:
        Choice-like values are detected heuristically and stored in
        `choice_value`, while free text is assembled in `answer_text`.
    """
    response_id_col = _find_column(consultation_data.columns, "Response ID")
    org_name_col = _find_column(consultation_data.columns, "4. What is your organisation name?")
    org_type_col = _find_column(
        consultation_data.columns,
        "6. Which category best describes your organisation? (Select all that apply) - Selected Choice",
    )
    region_col = _find_column(
        consultation_data.columns,
        "7. Which Nation or Region are you / your organisation located in, or interested in?",
    )

    output: list[ResponseItem] = []

    for row in consultation_data.rows:
        response_id = row.get(response_id_col.unique_name, "")
        organisation_name = row.get(org_name_col.unique_name, "Unknown organisation")
        organisation_type = row.get(org_type_col.unique_name, "")
        region = row.get(region_col.unique_name, "")

        for question in questions:
            primary_value = _clean_text(row.get(question.primary_column.unique_name, ""))
            supplemental_values = [
                _clean_text(row.get(col.unique_name, ""))
                for col in question.supplemental_columns
                if _clean_text(row.get(col.unique_name, ""))
            ]

            choice_value = primary_value if _looks_categorical(primary_value) else None
            text_parts: list[str] = []

            if primary_value and choice_value is None:
                text_parts.append(primary_value)
            text_parts.extend(supplemental_values)

            if choice_value and text_parts:
                answer_text = f"Choice: {choice_value}. " + " ".join(text_parts)
            elif choice_value:
                answer_text = choice_value
            else:
                answer_text = " ".join(text_parts)

            answer_text = _clean_text(answer_text)
            if not answer_text:
                continue

            excerpt = answer_text[:excerpt_chars].strip()
            if len(answer_text) > excerpt_chars:
                excerpt = f"{excerpt}..."

            record_id = f"{response_id}:{question.question_id}"
            output.append(
                ResponseItem(
                    record_id=record_id,
                    response_id=response_id,
                    organisation_name=organisation_name,
                    organisation_type=organisation_type,
                    region=region,
                    question_id=question.question_id,
                    question_text=question.question_text,
                    section=question.section,
                    choice_value=choice_value,
                    answer_text=answer_text,
                    excerpt=excerpt,
                )
            )

    return output


def list_organisations(prepared: PreparedData) -> list[tuple[str, str]]:
    """Return unique organisation options suitable for UI/CLI selection."""
    seen: set[str] = set()
    entries: list[tuple[str, str]] = []

    for row in prepared.consultation_data.rows:
        response_id = _row_value(prepared.consultation_data.columns, row, "Response ID")
        org_name = _row_value(prepared.consultation_data.columns, row, "4. What is your organisation name?")
        if not response_id or response_id in seen:
            continue
        seen.add(response_id)
        entries.append((response_id, f"{org_name} ({response_id})"))

    entries.sort(key=lambda pair: pair[1].lower())
    return entries


def get_question_options(prepared: PreparedData) -> list[tuple[str, str]]:
    """Return question options as `(question_id, display_label)` tuples."""
    return [(q.question_id, f"{q.question_id} | {q.question_text}") for q in prepared.questions]


def get_organisation_catalog(prepared: PreparedData, response_id: str) -> OrganisationCatalog:
    """Build the Approach 1 input object for a specific organisation.

    Inputs:
        prepared: Preprocessed dataset.
        response_id: Target submission identifier.

    Output:
        `OrganisationCatalog` including all answered response items.
    """
    items = [item for item in prepared.response_items if item.response_id == response_id]
    if not items:
        raise ValueError(f"No records found for response ID: {response_id}")

    first = items[0]
    answered = len({item.question_id for item in items})

    return OrganisationCatalog(
        response_id=response_id,
        organisation_name=first.organisation_name,
        organisation_type=first.organisation_type,
        region=first.region,
        answered_questions=answered,
        total_questions=len(prepared.questions),
        items=items,
    )


def get_question_slice(prepared: PreparedData, question_id: str) -> QuestionSlice:
    """Build the Approach 2 input slice for one question across organisations."""
    question = next((q for q in prepared.questions if q.question_id == question_id), None)
    if question is None:
        raise ValueError(f"Unknown question_id: {question_id}")

    items = [item for item in prepared.response_items if item.question_id == question_id]
    return QuestionSlice(question=question, items=items)


def calculate_distribution(items: list[ResponseItem]) -> dict[str, float]:
    """Compute percentage distribution for normalized categorical answers."""
    cleaned_values = [normalize_choice(item.choice_value) for item in items if item.choice_value]
    cleaned_values = [value for value in cleaned_values if value]

    if not cleaned_values:
        return {}

    counts = Counter(cleaned_values)
    total = sum(counts.values())
    return {label: round((count / total) * 100, 2) for label, count in counts.items()}


def normalize_choice(value: str | None) -> str:
    """Map variant raw choice text to canonical labels used in summaries."""
    if not value:
        return ""
    text = _clean_text(value).lower()

    normalized_aliases = {
        "strongly agree": "Strongly agree",
        "somewhat agree": "Somewhat agree",
        "neither agree nor disagree": "Neither agree nor disagree",
        "somewhat disagree": "Somewhat disagree",
        "strongly disagree": "Strongly disagree",
        "yes": "Yes",
        "no": "No",
        "maybe": "Maybe",
        "agree": "Agree",
        "disagree": "Disagree",
        "neutral": "Neutral",
        "no comment": "No comment",
    }

    for key, label in normalized_aliases.items():
        if text.startswith(key):
            return label

    return ""


def _find_question_start_index(columns: list[ColumnSpec]) -> int:
    """Locate the first consultation question column in the dataset."""
    for col in columns:
        if col.raw_name.startswith("1. Do you agree"):
            return col.index
    return 13


def _is_supplemental_header(lowered: str) -> bool:
    """Identify headers that carry free-text supplements for a primary question."""
    if " - yes - text" in lowered or " - maybe - text" in lowered or " - no - text" in lowered:
        return True
    return lowered.startswith(SUPPLEMENT_PREFIXES)


def _looks_categorical(value: str) -> bool:
    """Heuristically decide whether a cell value is a structured choice label."""
    if not value:
        return False

    lowered = value.lower().strip()
    if lowered in CATEGORICAL_HINTS:
        return True

    if len(lowered) <= 24 and lowered.replace("-", " ") in CATEGORICAL_HINTS:
        return True

    title_value = lowered.title()
    if title_value in {"Strongly Agree", "Somewhat Agree", "Somewhat Disagree", "Strongly Disagree"}:
        return True

    if len(lowered.split()) <= 3 and len(lowered) <= 25 and lowered.isalpha():
        return True

    return False


def _canonical_question_text(raw: str) -> str:
    """Normalize question headers into stable display/ID text."""
    text = re.sub(r"^\d+\.\s*", "", raw)
    text = text.replace(" - Selected Choice", "")
    return _clean_text(text)


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove hidden unicode markers."""
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_column(columns: list[ColumnSpec], startswith: str) -> ColumnSpec:
    """Find the first column whose raw header starts with the provided prefix."""
    for col in columns:
        if col.raw_name.startswith(startswith):
            return col
    raise KeyError(f"Column not found: {startswith}")


def _row_value(columns: list[ColumnSpec], row: dict[str, str], startswith: str) -> str:
    """Read and clean a row value for a required column prefix."""
    col = _find_column(columns, startswith)
    return _clean_text(row.get(col.unique_name, ""))


def load_section_mapping(columns: list[ColumnSpec], path: Path | None) -> dict[int, str]:
    """Load section mapping from XLSX and align it with CSV columns.

    Inputs:
        columns: CSV column definitions.
        path: Optional path to mapping workbook.

    Output:
        Dictionary keyed by column index with section names.

    Notes:
        Preferred alignment uses row order (mapping file is expected to mirror
        CSV header order). A header+occurrence fallback is used if row-order
        validation fails.
    """
    if path is None or not path.exists():
        return {}

    try:
        rows = _read_xlsx_rows(path)
    except Exception:
        return {}

    if len(rows) <= 1:
        return {}

    data_rows = rows[1:]
    if not data_rows:
        return {}

    mapping = _align_sections_by_index(columns, data_rows)
    if mapping:
        return mapping

    return _align_sections_by_header_occurrence(columns, data_rows)


def _align_sections_by_index(columns: list[ColumnSpec], data_rows: list[list[str]]) -> dict[int, str]:
    """Align section mapping by strict row order and exact header match."""
    if len(data_rows) < len(columns):
        return {}

    mapping: dict[int, str] = {}
    for col, row in zip(columns, data_rows):
        mapped_question = _clean_text(row[0] if row else "")
        if mapped_question != _clean_text(col.raw_name):
            return {}

        section = _clean_text(row[1] if len(row) > 1 else "")
        if section:
            mapping[col.index] = section

    return mapping


def _align_sections_by_header_occurrence(columns: list[ColumnSpec], data_rows: list[list[str]]) -> dict[int, str]:
    """Fallback alignment using `(header_text, occurrence_number)` keys."""
    from collections import defaultdict

    occ_map: dict[tuple[str, int], str] = {}
    row_occ: dict[str, int] = defaultdict(int)

    for row in data_rows:
        question = _clean_text(row[0] if row else "")
        section = _clean_text(row[1] if len(row) > 1 else "")
        if not question:
            continue
        row_occ[question] += 1
        occ_map[(question, row_occ[question])] = section

    out: dict[int, str] = {}
    col_occ: dict[str, int] = defaultdict(int)
    for col in columns:
        question = _clean_text(col.raw_name)
        col_occ[question] += 1
        section = _clean_text(occ_map.get((question, col_occ[question]), ""))
        if section:
            out[col.index] = section

    return out


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    """Read first worksheet rows from an XLSX file using stdlib XML parsing."""
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ns_pkg = "http://schemas.openxmlformats.org/package/2006/relationships"

    with ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive, ns_main)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{ns_pkg}}}Relationship")
        }

        first_sheet = workbook.find(f".//{{{ns_main}}}sheet")
        if first_sheet is None:
            return []

        rel_id = first_sheet.attrib.get(f"{{{ns_rel}}}id", "")
        target = rel_map.get(rel_id, "")
        if not target:
            return []

        sheet_xml = ET.fromstring(archive.read(f"xl/{target}"))
        rows: list[list[str]] = []
        for row in sheet_xml.findall(f".//{{{ns_main}}}sheetData/{{{ns_main}}}row"):
            values: list[str] = []
            for cell in row.findall(f"{{{ns_main}}}c"):
                values.append(_read_cell_value(cell, ns_main, shared_strings))
            rows.append(values)

    return rows


def _read_shared_strings(archive: ZipFile, ns_main: str) -> list[str]:
    """Read workbook shared strings table."""
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []

    root = ET.fromstring(archive.read(name))
    out: list[str] = []
    for item in root.findall(f"{{{ns_main}}}si"):
        text = "".join((t.text or "") for t in item.iter(f"{{{ns_main}}}t"))
        out.append(text)
    return out


def _read_cell_value(cell: ET.Element, ns_main: str, shared_strings: list[str]) -> str:
    """Read plain text value from one XLSX cell element."""
    cell_type = cell.attrib.get("t", "")
    value = cell.find(f"{{{ns_main}}}v")
    if value is not None:
        raw = value.text or ""
        if cell_type == "s":
            try:
                idx = int(raw)
                return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
            except ValueError:
                return ""
        return raw

    inline = cell.find(f"{{{ns_main}}}is")
    if inline is None:
        return ""
    return "".join((t.text or "") for t in inline.iter(f"{{{ns_main}}}t"))
