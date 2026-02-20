import re
from pathlib import Path

from neso_consultations.cache import SummaryCache
from neso_consultations.config import Settings
from neso_consultations.ingestion import load_consultation_csv
from neso_consultations.llm.base import LLMJsonResult, LLMProvider
from neso_consultations.models import LLMUsage
from neso_consultations.processing import get_question_options, prepare_data
from neso_consultations.service import ConsultationService


class FakeLLMProvider(LLMProvider):
    def complete_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> LLMJsonResult:
        """Return deterministic JSON payloads for offline pipeline tests."""
        record_ids = re.findall(r"[A-Za-z0-9_-]+:Q\d+", user_prompt)
        first_id = record_ids[0] if record_ids else "UNKNOWN:Q00"
        second_id = record_ids[1] if len(record_ids) > 1 else first_id

        if "main_points" in user_prompt and "Section:" in user_prompt:
            payload = {
                "main_points": [{"text": "Main point", "evidence_ids": [first_id]}],
                "concerns": [{"text": "Concern point", "evidence_ids": [second_id]}],
                "asks": [{"text": "Ask point", "evidence_ids": [first_id]}],
                "nuances": [{"text": "Nuance point", "evidence_ids": [second_id]}],
            }
        elif "hybrid organisation summary" in user_prompt:
            payload = {
                "overall_stance": "mixed",
                "key_supports": [{"text": "Support", "evidence_ids": [first_id]}],
                "key_concerns": [{"text": "Concern", "evidence_ids": [second_id]}],
                "asks_or_recommendations": [{"text": "Recommendation", "evidence_ids": [first_id]}],
            }
        else:
            payload = {
                "headline": "Question headline",
                "narrative": "Question narrative",
                # Deliberately omit evidence IDs to exercise enrichment fallback.
                "majority_view": [{"text": "Majority support for proposed approach"}],
                "minority_view": [{"text": "Minority concern around implementation risk"}],
                "key_arguments_for": [{"text": "Benefits of the proposal"}],
                "key_arguments_against": [{"text": "Potential downside and risk"}],
                "mainstream_clusters": [
                    {
                        "cluster_id": "C1",
                        "label": "Mainstream cluster",
                        "stance": "support",
                        "member_record_ids": [],
                        "evidence_ids": [],
                        "significance": "",
                    }
                ],
                "minority_clusters": [
                    {
                        "cluster_id": "M1",
                        "label": "Minority cluster",
                        "stance": "concern",
                        "member_record_ids": [],
                        "evidence_ids": [],
                        "significance": "",
                    }
                ],
            }

        return LLMJsonResult(payload=payload, usage=LLMUsage(input_tokens=100, output_tokens=50))


class TimeoutLLMProvider(LLMProvider):
    def complete_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> LLMJsonResult:
        raise TimeoutError("simulated timeout")


def _build_test_service(tmp_path: Path) -> ConsultationService:
    """Construct a service wired to fake LLM and temporary cache DB."""
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        data_path=root / "data" / "data.csv",
        section_mapping_path=root / "data" / "survey questrion-section mapping.xlsx",
        cache_path=tmp_path / "test_cache.sqlite",
        llm_provider="openai",
        openai_api_key="test",
        openai_model="fake-model",
        openai_base_url=None,
        azure_openai_endpoint="",
        azure_openai_api_version="2024-06-01",
        azure_openai_deployment="",
        azure_openai_api_key="",
        azure_openai_use_aad=False,
        azure_openai_managed_identity_client_id="",
        azure_openai_token_scope="https://cognitiveservices.azure.com/.default",
        llm_timeout_seconds=60,
        llm_max_retries=0,
        prompt_excerpt_chars=200,
        low_sample_threshold=8,
        high_missingness_threshold=0.35,
        input_cost_per_1k_tokens=0.001,
        output_cost_per_1k_tokens=0.002,
    )
    return ConsultationService(settings=settings, llm=FakeLLMProvider(), cache=SummaryCache(settings.cache_path))


def test_pipeline_generates_both_approaches(tmp_path: Path):
    """Ensure both summary approaches execute and return structured outputs."""
    service = _build_test_service(tmp_path)

    org_id = service.list_organisations()[0][0]
    org_result = service.summarise_organisation(response_id=org_id, use_cache=False)

    assert org_result.approach == "approach_1"
    assert org_result.section_summaries
    assert org_result.metrics.coverage > 0
    assert org_result.evidence_index

    question_id = get_question_options(service.prepared_data())[0][0]
    question_result = service.summarise_question(question_id=question_id, use_cache=False)

    assert question_result.approach == "approach_2"
    assert question_result.headline
    assert question_result.mainstream_clusters
    assert question_result.metrics.latency_seconds >= 0
    assert question_result.majority_view
    assert question_result.majority_view[0].count >= 1
    assert question_result.majority_view[0].evidence_ids
    assert question_result.mainstream_clusters[0].member_count >= 1
    assert question_result.mainstream_clusters[0].evidence_ids
    assert question_result.mainstream_clusters[0].description


def test_cache_roundtrip(tmp_path: Path):
    """Verify cache-backed calls return consistent organisation results."""
    service = _build_test_service(tmp_path)
    org_id = service.list_organisations()[0][0]

    first = service.summarise_organisation(response_id=org_id, use_cache=True)
    second = service.summarise_organisation(response_id=org_id, use_cache=True)

    assert first.response_id == second.response_id
    assert first.overall_stance == second.overall_stance


def test_question_timeout_fallback(tmp_path: Path):
    """Approach 2 should return a deterministic fallback when LLM times out."""
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        data_path=root / "data" / "data.csv",
        section_mapping_path=root / "data" / "survey questrion-section mapping.xlsx",
        cache_path=tmp_path / "timeout_cache.sqlite",
        llm_provider="openai",
        openai_api_key="test",
        openai_model="fake-model",
        openai_base_url=None,
        azure_openai_endpoint="",
        azure_openai_api_version="2024-06-01",
        azure_openai_deployment="",
        azure_openai_api_key="",
        azure_openai_use_aad=False,
        azure_openai_managed_identity_client_id="",
        azure_openai_token_scope="https://cognitiveservices.azure.com/.default",
        llm_timeout_seconds=60,
        llm_max_retries=0,
        prompt_excerpt_chars=200,
        low_sample_threshold=8,
        high_missingness_threshold=0.35,
        input_cost_per_1k_tokens=0.001,
        output_cost_per_1k_tokens=0.002,
    )
    service = ConsultationService(settings=settings, llm=TimeoutLLMProvider(), cache=SummaryCache(settings.cache_path))
    question_id = get_question_options(service.prepared_data())[0][0]
    result = service.summarise_question(question_id=question_id, use_cache=False)

    assert result.headline.startswith("Fallback summary")
    assert result.mainstream_clusters
    assert result.mainstream_clusters[0].member_count >= 1
    assert result.majority_view
    assert result.majority_view[0].evidence_ids
