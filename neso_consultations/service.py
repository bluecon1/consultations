from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from neso_consultations.cache import SummaryCache
from neso_consultations.config import Settings
from neso_consultations.ingestion import load_consultation_csv
from neso_consultations.llm.base import LLMProvider
from neso_consultations.models import (
    BulletPoint,
    EvidenceRef,
    OrganisationSummaryResult,
    PreparedData,
    QuestionCluster,
    QuestionSummaryResult,
    SectionSummary,
    SummaryMetrics,
    dataclass_to_dict,
)
from neso_consultations.processing import (
    get_organisation_catalog,
    get_question_options,
    get_question_slice,
    list_organisations,
    prepare_data,
)
from neso_consultations.summarisation import summarise_organisation, summarise_question


class ConsultationService:
    def __init__(self, *, settings: Settings, llm: LLMProvider, cache: SummaryCache) -> None:
        """Initialise the orchestration service used by CLI and UI layers.

        Inputs:
            settings: Runtime configuration and file/model paths.
            llm: Provider implementing `LLMProvider`.
            cache: Summary cache backend.
        """
        self._settings = settings
        self._llm = llm
        self._cache = cache
        self._prepared_data: PreparedData | None = None

    @property
    def settings(self) -> Settings:
        """Expose immutable runtime settings."""
        return self._settings

    def prepared_data(self) -> PreparedData:
        """Lazily load and preprocess source CSV data once per service instance.

        Output:
            `PreparedData` reused by list and summary operations.
        """
        if self._prepared_data is None:
            consultation_data = load_consultation_csv(self._settings.data_path)
            self._prepared_data = prepare_data(
                consultation_data,
                excerpt_chars=self._settings.prompt_excerpt_chars,
                section_mapping_path=self._settings.section_mapping_path,
            )
        return self._prepared_data

    def list_organisations(self) -> list[tuple[str, str]]:
        """Return selectable organisation options for clients."""
        return list_organisations(self.prepared_data())

    def list_questions(self) -> list[tuple[str, str]]:
        """Return selectable question options for clients."""
        return get_question_options(self.prepared_data())

    def summarise_organisation(self, *, response_id: str, use_cache: bool = True) -> OrganisationSummaryResult:
        """Generate or load a cached Approach 1 organisation summary.

        Inputs:
            response_id: Target submission ID.
            use_cache: Whether to attempt cache read/write.

        Output:
            `OrganisationSummaryResult`.
        """
        data = self.prepared_data()
        cache_key = self._cache.make_key(
            approach="approach_1",
            target_id=response_id,
            model=self._settings.model_identity,
            data_fingerprint=self._data_fingerprint(self._settings.data_path),
        )

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached:
                return _organisation_result_from_dict(cached)

        catalog = get_organisation_catalog(data, response_id)
        result = summarise_organisation(llm=self._llm, settings=self._settings, catalog=catalog)

        self._cache.set(cache_key, dataclass_to_dict(result))
        return result

    def summarise_question(self, *, question_id: str, use_cache: bool = True) -> QuestionSummaryResult:
        """Generate or load a cached Approach 2 question summary.

        Inputs:
            question_id: Target question identifier.
            use_cache: Whether to attempt cache read/write.

        Output:
            `QuestionSummaryResult`.
        """
        data = self.prepared_data()
        cache_key = self._cache.make_key(
            approach="approach_2",
            target_id=question_id,
            model=self._settings.model_identity,
            data_fingerprint=self._data_fingerprint(self._settings.data_path),
        )

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached:
                return _question_result_from_dict(cached)

        question_slice = get_question_slice(data, question_id)
        total_organisations = len({item.response_id for item in data.response_items})

        result = summarise_question(
            llm=self._llm,
            settings=self._settings,
            question_slice=question_slice,
            total_organisations=total_organisations,
        )

        self._cache.set(cache_key, dataclass_to_dict(result))
        return result

    @staticmethod
    def _data_fingerprint(path: Path) -> str:
        """Compute a short fingerprint for cache invalidation on data changes."""
        stat = path.stat()
        payload = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _organisation_result_from_dict(payload: dict[str, Any]) -> OrganisationSummaryResult:
    """Rehydrate an organisation summary dataclass from cached JSON-like dict."""
    return OrganisationSummaryResult(
        approach=str(payload.get("approach", "approach_1")),
        response_id=str(payload.get("response_id", "")),
        organisation_name=str(payload.get("organisation_name", "")),
        organisation_type=str(payload.get("organisation_type", "")),
        region=str(payload.get("region", "")),
        overall_stance=str(payload.get("overall_stance", "mixed")),
        key_supports=_bullets_from(payload.get("key_supports", [])),
        key_concerns=_bullets_from(payload.get("key_concerns", [])),
        asks_or_recommendations=_bullets_from(payload.get("asks_or_recommendations", [])),
        section_summaries=_sections_from(payload.get("section_summaries", [])),
        evidence_index=_evidence_from(payload.get("evidence_index", [])),
        metrics=_metrics_from(payload.get("metrics", {})),
    )


def _question_result_from_dict(payload: dict[str, Any]) -> QuestionSummaryResult:
    """Rehydrate a question summary dataclass from cached JSON-like dict."""
    return QuestionSummaryResult(
        approach=str(payload.get("approach", "approach_2")),
        question_id=str(payload.get("question_id", "")),
        question_text=str(payload.get("question_text", "")),
        section=str(payload.get("section", "")),
        headline=str(payload.get("headline", "")),
        narrative=str(payload.get("narrative", "")),
        majority_view=_bullets_from(payload.get("majority_view", [])),
        minority_view=_bullets_from(payload.get("minority_view", [])),
        key_arguments_for=_bullets_from(payload.get("key_arguments_for", [])),
        key_arguments_against=_bullets_from(payload.get("key_arguments_against", [])),
        distribution={str(k): float(v) for k, v in dict(payload.get("distribution", {})).items()},
        mainstream_clusters=_clusters_from(payload.get("mainstream_clusters", [])),
        minority_clusters=_clusters_from(payload.get("minority_clusters", [])),
        evidence_index=_evidence_from(payload.get("evidence_index", [])),
        metrics=_metrics_from(payload.get("metrics", {})),
    )


def _bullets_from(values: Any) -> list[BulletPoint]:
    """Parse a flexible list payload into validated `BulletPoint` objects."""
    if not isinstance(values, list):
        return []

    bullets: list[BulletPoint] = []
    for value in values:
        if isinstance(value, dict):
            text = str(value.get("text", "")).strip()
            evidence_ids = [str(v) for v in value.get("evidence_ids", []) if isinstance(v, (str, int))]
            count = int(value.get("count", 0) or 0)
            supporting_response_ids = [
                str(v) for v in value.get("supporting_response_ids", []) if isinstance(v, (str, int))
            ]
            supporting_organisations = [
                str(v) for v in value.get("supporting_organisations", []) if isinstance(v, str)
            ]
        else:
            text = str(value).strip()
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


def _sections_from(values: Any) -> list[SectionSummary]:
    """Parse serialized section summaries into typed `SectionSummary` objects."""
    if not isinstance(values, list):
        return []

    sections: list[SectionSummary] = []
    for value in values:
        if not isinstance(value, dict):
            continue

        sections.append(
            SectionSummary(
                section=str(value.get("section", "")),
                main_points=_bullets_from(value.get("main_points", [])),
                concerns=_bullets_from(value.get("concerns", [])),
                asks=_bullets_from(value.get("asks", [])),
                nuances=_bullets_from(value.get("nuances", [])),
                records_summarised=int(value.get("records_summarised", 0)),
                total_records=int(value.get("total_records", 0)),
            )
        )

    return sections


def _clusters_from(values: Any) -> list[QuestionCluster]:
    """Parse serialized cluster payloads into typed `QuestionCluster` objects."""
    if not isinstance(values, list):
        return []

    clusters: list[QuestionCluster] = []
    for value in values:
        if not isinstance(value, dict):
            continue

        clusters.append(
            QuestionCluster(
                cluster_id=str(value.get("cluster_id", "")),
                label=str(value.get("label", "")),
                stance=str(value.get("stance", "neutral")),
                member_record_ids=[str(v) for v in value.get("member_record_ids", []) if isinstance(v, (str, int))],
                evidence_ids=[str(v) for v in value.get("evidence_ids", []) if isinstance(v, (str, int))],
                significance=str(value.get("significance", "")),
                description=str(value.get("description", "")),
                member_count=int(value.get("member_count", 0)),
                response_count=int(value.get("response_count", 0)),
                organisation_count=int(value.get("organisation_count", 0)),
                supporting_response_ids=[
                    str(v) for v in value.get("supporting_response_ids", []) if isinstance(v, (str, int))
                ],
                supporting_organisations=[
                    str(v) for v in value.get("supporting_organisations", []) if isinstance(v, str)
                ],
            )
        )

    return clusters


def _evidence_from(values: Any) -> list[EvidenceRef]:
    """Parse serialized evidence entries into `EvidenceRef` objects."""
    if not isinstance(values, list):
        return []

    evidence: list[EvidenceRef] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        evidence.append(
            EvidenceRef(
                record_id=str(value.get("record_id", "")),
                excerpt=str(value.get("excerpt", "")),
            )
        )

    return evidence


def _metrics_from(value: Any) -> SummaryMetrics:
    """Parse serialized KPI payload into `SummaryMetrics` with safe defaults."""
    if not isinstance(value, dict):
        value = {}

    return SummaryMetrics(
        coverage=float(value.get("coverage", 0.0)),
        evidence_coverage=float(value.get("evidence_coverage", 0.0)),
        compression_ratio=float(value.get("compression_ratio", 0.0)),
        uncertainty_flags=[str(v) for v in value.get("uncertainty_flags", []) if isinstance(v, str)],
        latency_seconds=float(value.get("latency_seconds", 0.0)),
        cost_estimate_usd=float(value.get("cost_estimate_usd", 0.0)),
        input_chars=int(value.get("input_chars", 0)),
        output_chars=int(value.get("output_chars", 0)),
        input_tokens=int(value.get("input_tokens", 0)),
        output_tokens=int(value.get("output_tokens", 0)),
    )
