from __future__ import annotations

import json
import time
from collections import defaultdict

from neso_consultations.config import Settings
from neso_consultations.evaluation import build_metrics
from neso_consultations.llm.base import LLMProvider
from neso_consultations.models import (
    BulletPoint,
    LLMUsage,
    OrganisationCatalog,
    OrganisationSummaryResult,
    ResponseItem,
    SectionSummary,
)
from neso_consultations.summarisation.common import (
    build_evidence_index,
    detect_conflicting_signals,
    extract_referenced_ids_from_bullets,
    parse_bullets,
)


def summarise_organisation(
    *,
    llm: LLMProvider,
    settings: Settings,
    catalog: OrganisationCatalog,
) -> OrganisationSummaryResult:
    """Generate Approach 1 organisation-level hybrid summary.

    Inputs:
        llm: Provider used for section and roll-up generations.
        settings: Runtime thresholds and pricing assumptions.
        catalog: Organization-scoped response catalog.

    Output:
        `OrganisationSummaryResult` with section summaries, roll-up bullets,
        evidence index, and KPI metrics.

    Flow:
        1. Group responses by section.
        2. Summarize each section with evidence IDs.
        3. Roll up section summaries into an organization narrative.
        4. Build evidence index and compute metrics.
    """
    start = time.perf_counter()

    by_section: dict[str, list[ResponseItem]] = defaultdict(list)
    for item in catalog.items:
        by_section[item.section].append(item)

    total_usage = LLMUsage()
    total_input_chars = 0
    total_output_chars = 0

    section_summaries: list[SectionSummary] = []
    all_record_ids = {item.record_id for item in catalog.items}

    for section_name, section_items in by_section.items():
        payload, usage, input_chars, output_chars = _summarise_section(
            llm=llm,
            catalog=catalog,
            section_name=section_name,
            section_items=section_items,
        )

        total_usage = LLMUsage(
            input_tokens=total_usage.input_tokens + usage.input_tokens,
            output_tokens=total_usage.output_tokens + usage.output_tokens,
        )
        total_input_chars += input_chars
        total_output_chars += output_chars

        section_summaries.append(
            SectionSummary(
                section=section_name,
                main_points=parse_bullets(payload.get("main_points"), allowed_ids=all_record_ids),
                concerns=parse_bullets(payload.get("concerns"), allowed_ids=all_record_ids),
                asks=parse_bullets(payload.get("asks"), allowed_ids=all_record_ids),
                nuances=parse_bullets(payload.get("nuances"), allowed_ids=all_record_ids),
                records_summarised=len(section_items),
                total_records=len(section_items),
            )
        )

    rollup_payload, rollup_usage, rollup_input_chars, rollup_output_chars = _rollup_sections(
        llm=llm,
        catalog=catalog,
        section_summaries=section_summaries,
    )

    total_usage = LLMUsage(
        input_tokens=total_usage.input_tokens + rollup_usage.input_tokens,
        output_tokens=total_usage.output_tokens + rollup_usage.output_tokens,
    )
    total_input_chars += rollup_input_chars
    total_output_chars += rollup_output_chars

    key_supports = parse_bullets(rollup_payload.get("key_supports"), allowed_ids=all_record_ids)
    key_concerns = parse_bullets(rollup_payload.get("key_concerns"), allowed_ids=all_record_ids)
    asks = parse_bullets(rollup_payload.get("asks_or_recommendations"), allowed_ids=all_record_ids)

    all_bullets: list[BulletPoint] = [*key_supports, *key_concerns, *asks]
    for section in section_summaries:
        all_bullets.extend(section.main_points)
        all_bullets.extend(section.concerns)
        all_bullets.extend(section.asks)
        all_bullets.extend(section.nuances)

    referenced_ids = extract_referenced_ids_from_bullets(all_bullets)
    evidence_index = build_evidence_index(items=catalog.items, referenced_ids=referenced_ids)

    output_chars = len(json.dumps(rollup_payload, ensure_ascii=True)) + sum(
        len(json.dumps(section, default=_json_default, ensure_ascii=True)) for section in section_summaries
    )
    total_output_chars += output_chars

    metrics = build_metrics(
        coverage_numerator=catalog.answered_questions,
        coverage_denominator=catalog.total_questions,
        bullets=all_bullets,
        input_chars=total_input_chars,
        output_chars=total_output_chars,
        input_tokens=total_usage.input_tokens,
        output_tokens=total_usage.output_tokens,
        latency_seconds=time.perf_counter() - start,
        low_sample_threshold=settings.low_sample_threshold,
        high_missingness_threshold=settings.high_missingness_threshold,
        cost_per_1k_input=settings.input_cost_per_1k_tokens,
        cost_per_1k_output=settings.output_cost_per_1k_tokens,
        conflicting_signals=detect_conflicting_signals(catalog.items),
    )

    return OrganisationSummaryResult(
        approach="approach_1",
        response_id=catalog.response_id,
        organisation_name=catalog.organisation_name,
        organisation_type=catalog.organisation_type,
        region=catalog.region,
        overall_stance=str(rollup_payload.get("overall_stance", "mixed")).strip() or "mixed",
        key_supports=key_supports,
        key_concerns=key_concerns,
        asks_or_recommendations=asks,
        section_summaries=section_summaries,
        evidence_index=evidence_index,
        metrics=metrics,
    )


def _summarise_section(
    *,
    llm: LLMProvider,
    catalog: OrganisationCatalog,
    section_name: str,
    section_items: list[ResponseItem],
) -> tuple[dict, LLMUsage, int, int]:
    """Summarize one section for an organisation using source excerpts.

    Inputs:
        llm: LLM provider.
        catalog: Organisation metadata context.
        section_name: Name of the section being summarized.
        section_items: Response items for the section.

    Output:
        Tuple of `(payload, usage, input_chars, output_chars)`.
    """
    lines = []
    for item in section_items:
        lines.append(f"{item.record_id} | {item.question_text} | {item.excerpt}")

    user_prompt = (
        f"Organisation: {catalog.organisation_name}\n"
        f"Section: {section_name}\n"
        "Summarise the section. Preserve minority, conditional, and nuanced points.\n"
        "Source responses:\n"
        + "\n".join(lines)
        + "\n\nReturn JSON with keys: main_points, concerns, asks, nuances.\n"
        "Each key maps to a list of objects: {text, evidence_ids}.\n"
        "Use only record IDs provided above as evidence_ids."
    )

    system_prompt = (
        "You are a policy consultation summariser. Output valid JSON only. "
        "No markdown. No prose outside JSON."
    )

    result = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1)
    output_chars = len(json.dumps(result.payload, ensure_ascii=True))

    return result.payload, result.usage, len(user_prompt), output_chars


def _rollup_sections(
    *,
    llm: LLMProvider,
    catalog: OrganisationCatalog,
    section_summaries: list[SectionSummary],
) -> tuple[dict, LLMUsage, int, int]:
    """Produce final organisation roll-up from section-level summaries.

    Inputs:
        llm: LLM provider.
        catalog: Organisation metadata and coverage counts.
        section_summaries: Structured section outputs from `_summarise_section`.

    Output:
        Tuple of `(payload, usage, input_chars, output_chars)`.
    """
    section_payload = []
    for summary in section_summaries:
        section_payload.append(
            {
                "section": summary.section,
                "main_points": [b.text for b in summary.main_points],
                "concerns": [b.text for b in summary.concerns],
                "asks": [b.text for b in summary.asks],
                "nuances": [b.text for b in summary.nuances],
                "record_ids": sorted(
                    {
                        *[eid for b in summary.main_points for eid in b.evidence_ids],
                        *[eid for b in summary.concerns for eid in b.evidence_ids],
                        *[eid for b in summary.asks for eid in b.evidence_ids],
                        *[eid for b in summary.nuances for eid in b.evidence_ids],
                    }
                ),
            }
        )

    user_prompt = (
        f"Organisation: {catalog.organisation_name}\n"
        f"Type: {catalog.organisation_type}\n"
        f"Region: {catalog.region}\n"
        f"Answered questions: {catalog.answered_questions}/{catalog.total_questions}\n\n"
        "Create a hybrid organisation summary from section summaries.\n"
        "Preserve minority and nuanced points and include evidence IDs.\n"
        f"Section summaries JSON:\n{json.dumps(section_payload, ensure_ascii=True)}\n\n"
        "Return JSON with keys: overall_stance, key_supports, key_concerns, asks_or_recommendations.\n"
        "For bullet lists, each entry must be {text, evidence_ids}."
    )

    system_prompt = (
        "You summarise consultation responses. Output JSON only with explicit evidence linking. "
        "No extra keys."
    )

    result = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1)
    output_chars = len(json.dumps(result.payload, ensure_ascii=True))

    return result.payload, result.usage, len(user_prompt), output_chars


def _json_default(value: object) -> object:
    """Fallback JSON serializer used for output-size accounting."""
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
