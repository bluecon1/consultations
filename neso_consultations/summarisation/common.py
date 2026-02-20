from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from neso_consultations.models import BulletPoint, EvidenceRef, QuestionCluster, ResponseItem
from neso_consultations.processing import normalize_choice


@dataclass(frozen=True)
class TextStats:
    input_chars: int = 0
    output_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def parse_bullets(raw_value: Any, *, allowed_ids: set[str]) -> list[BulletPoint]:
    """Validate and normalize bullet structures returned by the model.

    Inputs:
        raw_value: Model payload value expected to be a list of bullets.
        allowed_ids: Record IDs allowed as evidence links.

    Output:
        Sanitized `BulletPoint` list with invalid evidence IDs removed.
    """
    if not isinstance(raw_value, list):
        return []

    bullets: list[BulletPoint] = []
    for item in raw_value:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            evidence_ids = [
                str(value)
                for value in item.get("evidence_ids", [])
                if isinstance(value, (str, int)) and str(value) in allowed_ids
            ]
            count = int(item.get("count", 0) or 0)
            supporting_response_ids = [
                str(value)
                for value in item.get("supporting_response_ids", [])
                if isinstance(value, (str, int))
            ]
            supporting_organisations = [
                str(value)
                for value in item.get("supporting_organisations", [])
                if isinstance(value, str)
            ]
        else:
            text = str(item).strip()
            evidence_ids = []
            count = 0
            supporting_response_ids = []
            supporting_organisations = []

        if text:
            bullets.append(
                BulletPoint(
                    text=text,
                    evidence_ids=evidence_ids,
                    count=count,
                    supporting_response_ids=supporting_response_ids,
                    supporting_organisations=supporting_organisations,
                )
            )

    return bullets


def parse_clusters(raw_value: Any, *, allowed_ids: set[str], fallback_prefix: str) -> list[QuestionCluster]:
    """Validate and normalize cluster structures returned by the model.

    Inputs:
        raw_value: Model payload value expected to be a list of clusters.
        allowed_ids: Record IDs allowed for members/evidence.
        fallback_prefix: Prefix used when cluster ID is missing.

    Output:
        Sanitized `QuestionCluster` list.
    """
    if not isinstance(raw_value, list):
        return []

    clusters: list[QuestionCluster] = []
    for idx, item in enumerate(raw_value, start=1):
        if not isinstance(item, dict):
            continue

        cluster_id = str(item.get("cluster_id", f"{fallback_prefix}_{idx}"))
        label = str(item.get("label", "")).strip()
        stance = str(item.get("stance", "neutral")).strip().lower() or "neutral"
        member_ids = [
            str(value)
            for value in item.get("member_record_ids", [])
            if isinstance(value, (str, int)) and str(value) in allowed_ids
        ]
        evidence_ids = [
            str(value)
            for value in item.get("evidence_ids", [])
            if isinstance(value, (str, int)) and str(value) in allowed_ids
        ]
        significance = str(item.get("significance", "")).strip()
        description = str(item.get("description", "")).strip()
        member_count = int(item.get("member_count", 0) or 0)
        response_count = int(item.get("response_count", 0) or 0)
        organisation_count = int(item.get("organisation_count", 0) or 0)
        supporting_response_ids = [
            str(value)
            for value in item.get("supporting_response_ids", [])
            if isinstance(value, (str, int))
        ]
        supporting_organisations = [
            str(value)
            for value in item.get("supporting_organisations", [])
            if isinstance(value, str)
        ]

        if not label:
            continue

        clusters.append(
            QuestionCluster(
                cluster_id=cluster_id,
                label=label,
                stance=stance,
                member_record_ids=member_ids,
                evidence_ids=evidence_ids,
                significance=significance,
                description=description,
                member_count=member_count,
                response_count=response_count,
                organisation_count=organisation_count,
                supporting_response_ids=supporting_response_ids,
                supporting_organisations=supporting_organisations,
            )
        )

    return clusters


def build_evidence_index(*, items: list[ResponseItem], referenced_ids: set[str]) -> list[EvidenceRef]:
    """Join referenced record IDs to local excerpts for evidence display."""
    if not referenced_ids:
        return []

    mapping = {item.record_id: item.excerpt for item in items}
    evidence = [
        EvidenceRef(record_id=record_id, excerpt=mapping[record_id])
        for record_id in sorted(referenced_ids)
        if record_id in mapping
    ]
    return evidence


def extract_referenced_ids_from_bullets(bullets: list[BulletPoint]) -> set[str]:
    """Collect all evidence IDs referenced across bullet points."""
    ids: set[str] = set()
    for bullet in bullets:
        ids.update(bullet.evidence_ids)
    return ids


def extract_referenced_ids_from_clusters(clusters: list[QuestionCluster]) -> set[str]:
    """Collect all member/evidence IDs referenced across clusters."""
    ids: set[str] = set()
    for cluster in clusters:
        ids.update(cluster.member_record_ids)
        ids.update(cluster.evidence_ids)
    return ids


def detect_conflicting_signals(items: list[ResponseItem]) -> bool:
    """Flag mixed stance signals where support and concern are both material.

    Input:
        items: Response items for one organisation or one question.

    Output:
        `True` when both support and concern ratios are at least 25%.
    """
    normalized = [normalize_choice(item.choice_value) for item in items if item.choice_value]
    counts = Counter(value for value in normalized if value)

    supportive = sum(counts.get(label, 0) for label in ["Strongly agree", "Somewhat agree", "Agree", "Yes"])
    concern = sum(counts.get(label, 0) for label in ["Strongly disagree", "Somewhat disagree", "Disagree", "No"])

    total = supportive + concern
    if total == 0:
        return False

    supportive_ratio = supportive / total
    concern_ratio = concern / total
    return supportive_ratio >= 0.25 and concern_ratio >= 0.25


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "our",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "what",
    "when",
    "where",
    "which",
    "would",
    "could",
    "should",
    "their",
    "them",
    "they",
    "about",
    "please",
    "provide",
    "reasoning",
    "approach",
    "agree",
    "disagree",
    "question",
    "response",
    "option",
    "page",
}


def enrich_bullets_with_support(
    bullets: list[BulletPoint],
    *,
    items: list[ResponseItem],
    default_top_k: int = 8,
) -> list[BulletPoint]:
    """Fill missing evidence/count metadata for viewpoint bullets."""
    record_map = {item.record_id: item for item in items}
    enriched: list[BulletPoint] = []

    for bullet in bullets:
        evidence_ids = [rid for rid in bullet.evidence_ids if rid in record_map]
        if not evidence_ids:
            evidence_ids = _match_record_ids_for_text(
                bullet.text,
                items=items,
                top_k=default_top_k,
            )

        supporting_items = [record_map[rid] for rid in evidence_ids if rid in record_map]
        response_ids = sorted({item.response_id for item in supporting_items})
        organisations = sorted({item.organisation_name for item in supporting_items})
        count = bullet.count or len(response_ids)

        enriched.append(
            BulletPoint(
                text=bullet.text,
                evidence_ids=evidence_ids,
                count=count,
                supporting_response_ids=response_ids,
                supporting_organisations=organisations,
            )
        )

    return enriched


def enrich_clusters_with_support(
    clusters: list[QuestionCluster],
    *,
    items: list[ResponseItem],
    fallback_prefix: str,
) -> list[QuestionCluster]:
    """Ensure cluster member/evidence/count fields are populated."""
    record_map = {item.record_id: item for item in items}
    out: list[QuestionCluster] = []

    source_clusters = clusters if clusters else build_fallback_clusters(items=items, prefix=fallback_prefix)

    for index, cluster in enumerate(source_clusters, start=1):
        member_ids = [rid for rid in cluster.member_record_ids if rid in record_map]

        if not member_ids:
            query_text = f"{cluster.label}. {cluster.significance or cluster.description}".strip()
            member_ids = _match_record_ids_for_text(query_text or cluster.label, items=items, top_k=14)
        if not member_ids:
            stance_bucket = [
                item.record_id
                for item in items
                if _stance_from_item(item) == (cluster.stance or "").lower()
            ]
            if stance_bucket:
                member_ids = stance_bucket[:14]
        if not member_ids and items:
            member_ids = [item.record_id for item in items[: min(8, len(items))]]

        evidence_ids = [rid for rid in cluster.evidence_ids if rid in record_map]
        if not evidence_ids:
            evidence_ids = member_ids[: min(8, len(member_ids))]

        member_items = [record_map[rid] for rid in member_ids if rid in record_map]
        response_ids = sorted({item.response_id for item in member_items})
        organisations = sorted({item.organisation_name for item in member_items})
        member_count = cluster.member_count or len(member_ids)
        response_count = cluster.response_count or len(response_ids)
        organisation_count = cluster.organisation_count or len(organisations)

        description = cluster.description or cluster.significance
        if not description:
            description = (
                f"{response_count} responses from {organisation_count} organisations "
                f"with {cluster.stance or 'mixed'} stance."
            )

        out.append(
            QuestionCluster(
                cluster_id=cluster.cluster_id or f"{fallback_prefix}_{index}",
                label=cluster.label or f"{fallback_prefix.title()} cluster {index}",
                stance=cluster.stance or "neutral",
                member_record_ids=member_ids,
                evidence_ids=evidence_ids,
                significance=cluster.significance,
                description=description,
                member_count=member_count,
                response_count=response_count,
                organisation_count=organisation_count,
                supporting_response_ids=response_ids,
                supporting_organisations=organisations,
            )
        )

    return out


def build_fallback_clusters(*, items: list[ResponseItem], prefix: str) -> list[QuestionCluster]:
    """Create deterministic stance-based clusters when model clusters are missing."""
    buckets: dict[str, list[ResponseItem]] = {
        "support": [],
        "concern": [],
        "neutral": [],
        "other": [],
    }

    for item in items:
        stance = _stance_from_item(item)
        buckets.setdefault(stance, []).append(item)

    clusters: list[QuestionCluster] = []
    ordered = sorted(buckets.items(), key=lambda pair: len(pair[1]), reverse=True)
    for idx, (stance, bucket_items) in enumerate(ordered, start=1):
        if not bucket_items:
            continue
        ids = [item.record_id for item in bucket_items]
        clusters.append(
            QuestionCluster(
                cluster_id=f"{prefix}_{idx}",
                label=f"{stance.title()} viewpoint",
                stance=stance,
                member_record_ids=ids,
                evidence_ids=ids[: min(8, len(ids))],
                significance=f"Auto-clustered by stance: {stance}.",
            )
        )
    return clusters


def _match_record_ids_for_text(text: str, *, items: list[ResponseItem], top_k: int) -> list[str]:
    query_tokens = _tokenize(text)
    if not query_tokens:
        return []

    scored: list[tuple[float, str]] = []
    for item in items:
        score = _token_overlap_score(query_tokens, _tokenize(item.answer_text))
        if score > 0:
            scored.append((score, item.record_id))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    min_score = 0.08
    selected = [rid for score, rid in scored if score >= min_score][:top_k]

    if not selected and scored:
        selected = [rid for _, rid in scored[: min(top_k, 3)]]

    return selected


def _tokenize(text: str) -> set[str]:
    parts = [
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split()
        if len(token) > 2 and token not in _STOPWORDS
    ]
    return set(parts)


def _token_overlap_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(len(query_tokens), 1)


def _stance_from_item(item: ResponseItem) -> str:
    normalized = normalize_choice(item.choice_value)
    support = {"Strongly agree", "Somewhat agree", "Agree", "Yes"}
    concern = {"Strongly disagree", "Somewhat disagree", "Disagree", "No"}
    neutral = {"Neither agree nor disagree", "Neutral", "Maybe", "No comment"}

    if normalized in support:
        return "support"
    if normalized in concern:
        return "concern"
    if normalized in neutral:
        return "neutral"

    text = (item.answer_text or "").lower()
    if any(word in text for word in ["support", "welcome", "agree"]):
        return "support"
    if any(word in text for word in ["concern", "risk", "oppose", "disagree"]):
        return "concern"
    return "other"
