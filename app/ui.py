from __future__ import annotations

import streamlit as st

from app.config import ensure_directories
from app.ingestion.ingest import ingest_corpus
from app.services.topic_to_draft import topic_to_draft


def main() -> None:
    ensure_directories()
    st.set_page_config(page_title="mebot", layout="wide")
    st.title("mebot")
    st.caption("Personal corpus-grounded drafting assistant")

    with st.sidebar:
        st.subheader("Corpus")
        if st.button("Run Ingestion"):
            result = ingest_corpus()
            st.success(
                f"Ingested {result.documents_ingested} documents into the local index "
                f"with {result.chunks_created} chunks."
            )

    topic = st.text_area("Topic", placeholder="零预算让旅行更有趣:读书和播客")
    requested_format = st.selectbox(
        "Format",
        options=["free-form", "newsletter section", "short essay", "xhs post"],
        index=0,
    )

    if st.button("Generate Draft", type="primary"):
        response = topic_to_draft(topic=topic, requested_format=requested_format)
        left, right = st.columns([1.5, 1])

        with left:
            st.subheader("Draft")
            st.write(response.draft_text)
            if response.abstained:
                st.info("Generation abstained because retrieval confidence was too weak.")
                if response.adjacent_topics:
                    st.caption("Adjacent topics from your corpus")
                    for suggestion in response.adjacent_topics:
                        st.write(f"- {suggestion}")
            with st.expander("Prompt Context Preview"):
                st.code(response.prompt_context_preview or "No prompt context.", language="text")

        with right:
            st.subheader("Sources")
            st.caption(response.retrieval_summary)
            if not response.sources:
                st.write("No source material found.")
            for source in response.sources:
                st.markdown(
                    f"**{source.title}**  \n"
                    f"{source.source_type} | {source.date_published or 'undated'} | score {source.score:.3f}"
                )
                if source.canonical_url:
                    st.write(source.canonical_url)
                for index, excerpt in enumerate(source.excerpts, start=1):
                    st.caption(f"Excerpt {index}")
                    st.write(excerpt)
                st.divider()


if __name__ == "__main__":
    main()
