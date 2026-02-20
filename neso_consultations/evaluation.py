from __future__ import annotations

from neso_consultations.models import BulletPoint, SummaryMetrics


def build_metrics(
    *,
    coverage_numerator: int,
    coverage_denominator: int,
    bullets: list[BulletPoint],
    input_chars: int,
    output_chars: int,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    low_sample_threshold: int,
    high_missingness_threshold: float,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
    conflicting_signals: bool,
) -> SummaryMetrics:
    """Compute deterministic quality and operational KPIs for a summary run.

    Inputs:
        coverage_numerator/coverage_denominator: Scope covered by the summary.
        bullets: Output bullet points (used for evidence-link coverage).
        input/output chars and tokens: Compression and cost accounting inputs.
        latency_seconds: End-to-end runtime for the summary call.
        thresholds: Values used for uncertainty flagging.
        cost_per_1k_*: Token pricing assumptions.
        conflicting_signals: Precomputed stance-conflict indicator.

    Output:
        `SummaryMetrics` populated with rounded values and uncertainty flags.
    """
    coverage = _ratio(coverage_numerator, coverage_denominator)

    bullet_count = len(bullets)
    with_evidence = sum(1 for b in bullets if b.evidence_ids)
    evidence_coverage = _ratio(with_evidence, bullet_count)

    compression_ratio = round(input_chars / max(output_chars, 1), 3)
    missingness = 1.0 - coverage

    flags: list[str] = []
    if coverage_numerator < low_sample_threshold:
        flags.append("low_sample_size")
    if conflicting_signals:
        flags.append("conflicting_stance_signals")
    if missingness >= high_missingness_threshold:
        flags.append("high_missingness")

    cost_estimate = (
        (input_tokens / 1000.0) * cost_per_1k_input
        + (output_tokens / 1000.0) * cost_per_1k_output
    )

    return SummaryMetrics(
        coverage=coverage,
        evidence_coverage=evidence_coverage,
        compression_ratio=compression_ratio,
        uncertainty_flags=flags,
        latency_seconds=round(latency_seconds, 3),
        cost_estimate_usd=round(cost_estimate, 6),
        input_chars=input_chars,
        output_chars=output_chars,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _ratio(numerator: int, denominator: int) -> float:
    """Return a safe rounded ratio in [0, 1], guarding divide-by-zero."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)
