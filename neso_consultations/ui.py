from __future__ import annotations

import streamlit as st

from neso_consultations.cli import build_service
from neso_consultations.models import BulletPoint, OrganisationSummaryResult, QuestionSummaryResult


st.set_page_config(page_title="NESO Consultation Summaries", layout="wide")


@st.cache_resource
def _get_service():
    """Create and cache a shared service instance for the Streamlit session."""
    return build_service()


def _render_bullets(title: str, bullets: list[BulletPoint]) -> None:
    """Render a labeled bullet list with inline evidence references."""
    st.markdown(f"**{title}**")
    if not bullets:
        st.write("- None")
        return

    for bullet in bullets:
        refs = ", ".join(bullet.evidence_ids) if bullet.evidence_ids else "no evidence"
        count_label = bullet.count if bullet.count else len(bullet.supporting_response_ids)
        responses = ", ".join(bullet.supporting_response_ids) if bullet.supporting_response_ids else "n/a"
        organisations = (
            ", ".join(bullet.supporting_organisations) if bullet.supporting_organisations else "n/a"
        )
        st.write(f"- {bullet.text}")
        st.caption(
            f"count={count_label} | responses={responses} | organisations={organisations} | evidence={refs}"
        )


def _render_metrics(result_metrics) -> None:
    """Render KPI cards and uncertainty flags for a summary result."""
    st.markdown("**KPIs**")
    cols = st.columns(6)
    cols[0].metric("Coverage", f"{result_metrics.coverage:.1%}")
    cols[1].metric("Evidence coverage", f"{result_metrics.evidence_coverage:.1%}")
    cols[2].metric("Compression", f"{result_metrics.compression_ratio:.2f}x")
    cols[3].metric("Latency (s)", f"{result_metrics.latency_seconds:.2f}")
    cols[4].metric("Cost est. (USD)", f"${result_metrics.cost_estimate_usd:.5f}")
    cols[5].metric("Prompt tokens", f"{result_metrics.input_tokens}")

    if result_metrics.uncertainty_flags:
        st.warning("Uncertainty flags: " + ", ".join(result_metrics.uncertainty_flags))


def _render_evidence_table(evidence_index) -> None:
    """Render evidence records (record ID + excerpt) as a table."""
    st.markdown("**Evidence Index**")
    if not evidence_index:
        st.info("No evidence references returned.")
        return

    st.dataframe(
        [{"record_id": ev.record_id, "excerpt": ev.excerpt} for ev in evidence_index],
        use_container_width=True,
        hide_index=True,
    )


def _render_approach_1(result: OrganisationSummaryResult) -> None:
    """Render the full Approach 1 output view in the UI."""
    st.subheader("Approach 1: Organisation Summary")
    st.write(
        f"**Organisation:** {result.organisation_name}  \\\n**Type:** {result.organisation_type or 'N/A'}  \\\n**Region:** {result.region or 'N/A'}  \\\n**Overall stance:** {result.overall_stance}"
    )

    _render_bullets("Key supports", result.key_supports)
    _render_bullets("Key concerns", result.key_concerns)
    _render_bullets("Asks / recommendations", result.asks_or_recommendations)

    st.markdown("**Section Summaries**")
    for section in result.section_summaries:
        with st.expander(f"{section.section} ({section.records_summarised} records)"):
            _render_bullets("Main points", section.main_points)
            _render_bullets("Concerns", section.concerns)
            _render_bullets("Asks", section.asks)
            _render_bullets("Nuances", section.nuances)

    _render_evidence_table(result.evidence_index)
    _render_metrics(result.metrics)


def _render_approach_2(result: QuestionSummaryResult) -> None:
    """Render the full Approach 2 output view in the UI."""
    st.subheader("Approach 2: Question Summary")
    st.write(
        f"**Question:** {result.question_id}  \\\n{result.question_text}  \\\n**Section:** {result.section}"
    )

    st.markdown(f"**Headline:** {result.headline}")
    st.write(result.narrative or "")

    if result.distribution:
        st.markdown("**Distribution**")
        st.bar_chart(result.distribution)

    _render_bullets("Majority view", result.majority_view)
    _render_bullets("Minority / edge view", result.minority_view)
    _render_bullets("Key arguments for", result.key_arguments_for)
    _render_bullets("Key arguments against", result.key_arguments_against)

    st.markdown("**Mainstream clusters**")
    for cluster in result.mainstream_clusters:
        description = cluster.description or cluster.significance or "No description provided."
        responses = ", ".join(cluster.supporting_response_ids) if cluster.supporting_response_ids else "n/a"
        organisations = (
            ", ".join(cluster.supporting_organisations) if cluster.supporting_organisations else "n/a"
        )
        st.write(
            f"- {cluster.cluster_id}: {cluster.label} ({cluster.stance}) | "
            f"members={cluster.member_count or len(cluster.member_record_ids)} | "
            f"responses={cluster.response_count} | organisations={cluster.organisation_count}"
        )
        st.caption(
            f"description={description} | responses={responses} | organisations={organisations} | "
            f"evidence={', '.join(cluster.evidence_ids) if cluster.evidence_ids else 'n/a'}"
        )

    st.markdown("**Minority clusters**")
    for cluster in result.minority_clusters:
        description = cluster.description or cluster.significance or "No description provided."
        responses = ", ".join(cluster.supporting_response_ids) if cluster.supporting_response_ids else "n/a"
        organisations = (
            ", ".join(cluster.supporting_organisations) if cluster.supporting_organisations else "n/a"
        )
        st.write(
            f"- {cluster.cluster_id}: {cluster.label} ({cluster.stance}) | "
            f"members={cluster.member_count or len(cluster.member_record_ids)} | "
            f"responses={cluster.response_count} | organisations={cluster.organisation_count}"
        )
        st.caption(
            f"description={description} | responses={responses} | organisations={organisations} | "
            f"evidence={', '.join(cluster.evidence_ids) if cluster.evidence_ids else 'n/a'}"
        )

    _render_evidence_table(result.evidence_index)
    _render_metrics(result.metrics)


def main() -> None:
    """Streamlit page entrypoint with two tabs for both approaches."""
    st.title("NESO Consultation Summaries")
    st.caption("Local-first summarisation with OpenAI, evidence linking, and KPI reporting")

    try:
        service = _get_service()
    except Exception as exc:
        st.error(f"Failed to initialise service: {exc}")
        st.info("Set OPENAI_API_KEY in .env, then restart the app.")
        return

    tab1, tab2 = st.tabs(["Approach 1: Organisation", "Approach 2: Question"])

    with tab1:
        organisations = service.list_organisations()
        if not organisations:
            st.error("No organisations found in CSV.")
        else:
            lookup = {label: response_id for response_id, label in organisations}
            selected_label = st.selectbox("Select organisation", list(lookup.keys()))
            use_cache = st.checkbox("Use cache", value=True, key="org_cache")

            if st.button("Generate organisation summary"):
                with st.spinner("Generating summary..."):
                    result = service.summarise_organisation(
                        response_id=lookup[selected_label],
                        use_cache=use_cache,
                    )
                _render_approach_1(result)

    with tab2:
        questions = service.list_questions()
        if not questions:
            st.error("No questions found in CSV.")
        else:
            lookup = {label: question_id for question_id, label in questions}
            selected_label = st.selectbox("Select question", list(lookup.keys()))
            use_cache = st.checkbox("Use cache", value=True, key="q_cache")

            if st.button("Generate question summary"):
                with st.spinner("Generating summary..."):
                    result = service.summarise_question(
                        question_id=lookup[selected_label],
                        use_cache=use_cache,
                    )
                _render_approach_2(result)


if __name__ == "__main__":
    main()
