from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ColumnSpec:
    unique_name: str
    raw_name: str
    index: int


@dataclass(frozen=True)
class QuestionDefinition:
    question_id: str
    question_text: str
    section: str
    primary_column: ColumnSpec
    supplemental_columns: list[ColumnSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ResponseItem:
    record_id: str
    response_id: str
    organisation_name: str
    organisation_type: str
    region: str
    question_id: str
    question_text: str
    section: str
    choice_value: str | None
    answer_text: str
    excerpt: str


@dataclass(frozen=True)
class OrganisationCatalog:
    response_id: str
    organisation_name: str
    organisation_type: str
    region: str
    answered_questions: int
    total_questions: int
    items: list[ResponseItem]


@dataclass(frozen=True)
class EvidenceRef:
    record_id: str
    excerpt: str


@dataclass(frozen=True)
class BulletPoint:
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    count: int = 0
    supporting_response_ids: list[str] = field(default_factory=list)
    supporting_organisations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SectionSummary:
    section: str
    main_points: list[BulletPoint] = field(default_factory=list)
    concerns: list[BulletPoint] = field(default_factory=list)
    asks: list[BulletPoint] = field(default_factory=list)
    nuances: list[BulletPoint] = field(default_factory=list)
    records_summarised: int = 0
    total_records: int = 0


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return total tokens consumed by one LLM interaction."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class SummaryMetrics:
    coverage: float
    evidence_coverage: float
    compression_ratio: float
    uncertainty_flags: list[str]
    latency_seconds: float
    cost_estimate_usd: float
    input_chars: int
    output_chars: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class OrganisationSummaryResult:
    approach: str
    response_id: str
    organisation_name: str
    organisation_type: str
    region: str
    overall_stance: str
    key_supports: list[BulletPoint]
    key_concerns: list[BulletPoint]
    asks_or_recommendations: list[BulletPoint]
    section_summaries: list[SectionSummary]
    evidence_index: list[EvidenceRef]
    metrics: SummaryMetrics


@dataclass(frozen=True)
class QuestionCluster:
    cluster_id: str
    label: str
    stance: str
    member_record_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    significance: str = ""
    description: str = ""
    member_count: int = 0
    response_count: int = 0
    organisation_count: int = 0
    supporting_response_ids: list[str] = field(default_factory=list)
    supporting_organisations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QuestionSummaryResult:
    approach: str
    question_id: str
    question_text: str
    section: str
    headline: str
    narrative: str
    majority_view: list[BulletPoint]
    minority_view: list[BulletPoint]
    key_arguments_for: list[BulletPoint]
    key_arguments_against: list[BulletPoint]
    distribution: dict[str, float]
    mainstream_clusters: list[QuestionCluster]
    minority_clusters: list[QuestionCluster]
    evidence_index: list[EvidenceRef]
    metrics: SummaryMetrics


@dataclass(frozen=True)
class ConsultationData:
    columns: list[ColumnSpec]
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class PreparedData:
    consultation_data: ConsultationData
    questions: list[QuestionDefinition]
    response_items: list[ResponseItem]


def dataclass_to_dict(value: Any) -> Any:
    """Convert a dataclass value into a plain dictionary.

    Input:
        Any Python object, typically one of this module's dataclasses.

    Output:
        `dict` for dataclass objects, otherwise the original input value.
    """
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
