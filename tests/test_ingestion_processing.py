from pathlib import Path

from neso_consultations.ingestion import load_consultation_csv
from neso_consultations.processing import list_organisations, prepare_data


def test_csv_load_and_prepare():
    """Validate CSV ingestion and preprocessing assumptions on sample data."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "data.csv"
    consultation_data = load_consultation_csv(data_path)

    assert len(consultation_data.rows) > 200
    assert len(consultation_data.columns) == 69

    duplicate_reason_cols = [
        c.unique_name for c in consultation_data.columns if c.raw_name == "Please provide your reasoning?"
    ]
    assert len(duplicate_reason_cols) >= 2

    section_mapping_path = (
        Path(__file__).resolve().parents[1] / "data" / "survey questrion-section mapping.xlsx"
    )
    prepared = prepare_data(
        consultation_data,
        excerpt_chars=180,
        section_mapping_path=section_mapping_path,
    )
    assert len(prepared.questions) >= 20
    assert len(prepared.response_items) > 1000
    sections = {q.section for q in prepared.questions}
    assert "Governance" in sections
    assert "Pathways" in sections

    orgs = list_organisations(prepared)
    assert len(orgs) > 150
    assert orgs[0][0]
