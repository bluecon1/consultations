from __future__ import annotations

import json
import time

from neso_consultations.config import Settings
from neso_consultations.evaluation import build_metrics
from neso_consultations.llm.base import LLMProvider
from neso_consultations.models import BulletPoint, LLMUsage, QuestionCluster, QuestionSummaryResult
from neso_consultations.processing import QuestionSlice, calculate_distribution
from neso_consultations.summarisation.common import (
    build_evidence_index,
    detect_conflicting_signals,
    enrich_bullets_with_support,
    enrich_clusters_with_support,
    extract_referenced_ids_from_bullets,
    extract_referenced_ids_from_clusters,
    parse_bullets,
    parse_clusters,
)


def summarise_question(
    *,
    llm: LLMProvider,
    settings: Settings,
    question_slice: QuestionSlice,
    total_organisations: int,
) -> QuestionSummaryResult:
    """Generate Approach 2 question-level cross-organisation summary.

    Inputs:
        llm: Provider used for synthesis and clustering narrative output.
        settings: Runtime thresholds and pricing assumptions.
        question_slice: All responses for one question plus question metadata.
        total_organisations: Denominator for coverage KPI.

    Output:
        `QuestionSummaryResult` including distribution, majority/minority
        arguments, clusters, evidence index, and KPI metrics.
    """
    start = time.perf_counter()

    distribution = calculate_distribution(question_slice.items)
    allowed_ids = {item.record_id for item in question_slice.items}

    response_lines = []
    for item in question_slice.items:
        choice = item.choice_value or ""
        response_lines.append(
            f"{item.record_id} | {item.organisation_name} | {choice} | {item.excerpt}"
        )

    user_prompt = (
        f"Question ID: {question_slice.question.question_id}\n"
        f"Question text: {question_slice.question.question_text}\n"
        f"Section: {question_slice.question.section}\n"
        f"Distribution (if available): {json.dumps(distribution, ensure_ascii=True)}\n"
        "Summarise claims, cluster mainstream positions, capture minority/outlier views, and include evidence IDs.\n"
        "Responses:\n"
        + "\n".join(response_lines)
        + "\n\nReturn JSON with keys:\n"
        "headline (str), narrative (str), majority_view (list), minority_view (list), "
        "key_arguments_for (list), key_arguments_against (list), mainstream_clusters (list), minority_clusters (list).\n"
        "For list bullets: [{text, evidence_ids}]\n"
        "For clusters: [{cluster_id, label, stance, member_record_ids, evidence_ids, significance}]\n"
        "Use only record IDs from the provided responses."
    )

    system_prompt = (
        "You summarise policy consultation responses across organisations. "
        "Preserve minority perspectives. Output valid JSON only."
    )

    try:
        result = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1)
        payload = result.payload
    except Exception as exc:
        # Graceful fallback for transient API/network failures.
        payload = _fallback_payload(question_slice=question_slice, distribution=distribution, error=exc)
        result = LLMUsage(input_tokens=0, output_tokens=0)

    majority_view = parse_bullets(payload.get("majority_view"), allowed_ids=allowed_ids)
    minority_view = parse_bullets(payload.get("minority_view"), allowed_ids=allowed_ids)
    key_for = parse_bullets(payload.get("key_arguments_for"), allowed_ids=allowed_ids)
    key_against = parse_bullets(payload.get("key_arguments_against"), allowed_ids=allowed_ids)

    # Enrich viewpoint bullets with deterministic evidence/count metadata.
    majority_view = enrich_bullets_with_support(majority_view, items=question_slice.items)
    minority_view = enrich_bullets_with_support(minority_view, items=question_slice.items)
    key_for = enrich_bullets_with_support(key_for, items=question_slice.items)
    key_against = enrich_bullets_with_support(key_against, items=question_slice.items)

    mainstream_clusters = parse_clusters(
        payload.get("mainstream_clusters"), allowed_ids=allowed_ids, fallback_prefix="mainstream"
    )
    minority_clusters = parse_clusters(
        payload.get("minority_clusters"), allowed_ids=allowed_ids, fallback_prefix="minority"
    )
    mainstream_clusters = enrich_clusters_with_support(
        mainstream_clusters,
        items=question_slice.items,
        fallback_prefix="mainstream",
    )
    minority_clusters = enrich_clusters_with_support(
        minority_clusters,
        items=question_slice.items,
        fallback_prefix="minority",
    )

    if not majority_view and mainstream_clusters:
        cluster = mainstream_clusters[0]
        majority_view = [
            _bullet_from_cluster(cluster, fallback_text=f"Mainstream view: {cluster.label}")
        ]
    if not minority_view and minority_clusters:
        cluster = minority_clusters[0]
        minority_view = [
            _bullet_from_cluster(cluster, fallback_text=f"Minority view: {cluster.label}")
        ]
    if not key_for:
        support_cluster = next((c for c in mainstream_clusters if c.stance == "support"), None)
        if support_cluster:
            key_for = [_bullet_from_cluster(support_cluster, fallback_text=support_cluster.label)]
    if not key_against:
        concern_cluster = next((c for c in minority_clusters if c.stance == "concern"), None)
        if concern_cluster:
            key_against = [_bullet_from_cluster(concern_cluster, fallback_text=concern_cluster.label)]

    referenced_ids = set()
    referenced_ids |= extract_referenced_ids_from_bullets(majority_view)
    referenced_ids |= extract_referenced_ids_from_bullets(minority_view)
    referenced_ids |= extract_referenced_ids_from_bullets(key_for)
    referenced_ids |= extract_referenced_ids_from_bullets(key_against)
    referenced_ids |= extract_referenced_ids_from_clusters(mainstream_clusters)
    referenced_ids |= extract_referenced_ids_from_clusters(minority_clusters)

    evidence_index = build_evidence_index(items=question_slice.items, referenced_ids=referenced_ids)

    all_bullets = [*majority_view, *minority_view, *key_for, *key_against]

    output_chars = len(json.dumps(payload, ensure_ascii=True))

    metrics = build_metrics(
        coverage_numerator=len(question_slice.items),
        coverage_denominator=total_organisations,
        bullets=all_bullets,
        input_chars=len(user_prompt),
        output_chars=output_chars,
        input_tokens=result.input_tokens if isinstance(result, LLMUsage) else result.usage.input_tokens,
        output_tokens=result.output_tokens if isinstance(result, LLMUsage) else result.usage.output_tokens,
        latency_seconds=time.perf_counter() - start,
        low_sample_threshold=settings.low_sample_threshold,
        high_missingness_threshold=settings.high_missingness_threshold,
        cost_per_1k_input=settings.input_cost_per_1k_tokens,
        cost_per_1k_output=settings.output_cost_per_1k_tokens,
        conflicting_signals=detect_conflicting_signals(question_slice.items),
    )

    return QuestionSummaryResult(
        approach="approach_2",
        question_id=question_slice.question.question_id,
        question_text=question_slice.question.question_text,
        section=question_slice.question.section,
        headline=str(payload.get("headline", "")).strip(),
        narrative=str(payload.get("narrative", "")).strip(),
        majority_view=majority_view,
        minority_view=minority_view,
        key_arguments_for=key_for,
        key_arguments_against=key_against,
        distribution=distribution,
        mainstream_clusters=mainstream_clusters,
        minority_clusters=minority_clusters,
        evidence_index=evidence_index,
        metrics=metrics,
    )


def _bullet_from_cluster(cluster: QuestionCluster, *, fallback_text: str) -> BulletPoint:
    text = cluster.description or cluster.significance or fallback_text
    return BulletPoint(
        text=text,
        evidence_ids=cluster.evidence_ids,
        count=cluster.response_count or cluster.member_count,
        supporting_response_ids=cluster.supporting_response_ids,
        supporting_organisations=cluster.supporting_organisations,
    )


def _fallback_payload(*, question_slice: QuestionSlice, distribution: dict[str, float], error: Exception) -> dict:
    """Build deterministic payload when LLM call fails (e.g. timeout)."""
    sorted_dist = sorted(distribution.items(), key=lambda pair: pair[1], reverse=True)
    if sorted_dist:
        dominant_label, dominant_pct = sorted_dist[0]
        headline = (
            f"Fallback summary (LLM timeout): dominant stance is {dominant_label} "
            f"at {dominant_pct:.1f}%."
        )
    else:
        headline = "Fallback summary (LLM timeout): no structured distribution available."

    narrative = (
        f"Generated without model response due to: {type(error).__name__}. "
        "Viewpoints and clusters are inferred from local response signals."
    )

    return {
        "headline": headline,
        "narrative": narrative,
        "majority_view": [],
        "minority_view": [],
        "key_arguments_for": [],
        "key_arguments_against": [],
        "mainstream_clusters": [],
        "minority_clusters": [],
    }
