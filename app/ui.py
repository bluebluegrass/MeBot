from __future__ import annotations

import streamlit as st

from app.config import ensure_directories
from app.ingestion.ingest import ingest_corpus
from app.services.topic_to_draft import draft_from_selected, retrieve_for_review


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

    topic = st.text_area(
        "你想写什么？",
        placeholder="我想用我在乌兹别克斯坦的游记写一篇小红书，主要是我突然去到乌兹别克的乡村去给小朋友们上雅思课，表达一段奇遇。",
        height=120,
    )
    requested_format = st.selectbox(
        "Format",
        options=["free-form", "newsletter section", "short essay", "xhs post"],
        index=0,
    )

    if st.button("Find Passages", type="primary"):
        # Clear stale state from a previous run
        for key in list(st.session_state.keys()):
            if key.startswith("include_source_"):
                del st.session_state[key]
        st.session_state["sources"] = retrieve_for_review(topic)
        st.session_state["topic"] = topic
        st.session_state["requested_format"] = requested_format
        st.session_state.pop("draft_response", None)

    # ── Step 2: passage selection ──────────────────────────────────────────────
    if st.session_state.get("sources") is not None:
        sources = st.session_state["sources"]

        if not sources:
            st.warning("没有找到相关语料。试试换一种描述方式？")
        else:
            st.subheader(f"找到 {len(sources)} 篇相关文章，选择你想用的段落：")
            for i, source in enumerate(sources):
                with st.expander(
                    f"{'✅' if st.session_state.get(f'include_source_{i}', True) else '☐'} "
                    f"{source.title}  —  score {source.score:.3f}",
                    expanded=True,
                ):
                    st.caption(
                        f"{source.source_type} | {source.date_published or 'undated'}"
                        + (f" | {source.canonical_url}" if source.canonical_url else "")
                    )
                    for j, excerpt in enumerate(source.excerpts, start=1):
                        st.caption(f"Passage {j}")
                        st.write(excerpt)
                    st.checkbox(
                        "用这篇",
                        value=True,
                        key=f"include_source_{i}",
                    )

            if st.button("Generate Draft", type="primary"):
                selected = [
                    sources[i]
                    for i in range(len(sources))
                    if st.session_state.get(f"include_source_{i}", True)
                ]
                st.session_state["draft_response"] = draft_from_selected(
                    st.session_state["topic"],
                    selected,
                    st.session_state["requested_format"],
                )

    # ── Step 3: draft output ───────────────────────────────────────────────────
    if st.session_state.get("draft_response"):
        response = st.session_state["draft_response"]
        st.divider()
        left, right = st.columns([1.5, 1])

        with left:
            st.subheader("Draft")
            st.write(response.draft_text)
            with st.expander("Prompt Context Preview"):
                st.code(response.prompt_context_preview or "No prompt context.", language="text")

        with right:
            st.subheader("Sources used")
            st.caption(response.retrieval_summary)
            if not response.sources:
                st.write("No source material.")
            for source in response.sources:
                st.markdown(
                    f"**{source.title}**  \n"
                    f"{source.source_type} | {source.date_published or 'undated'} | score {source.score:.3f}"
                )
                if source.canonical_url:
                    st.write(source.canonical_url)
                st.divider()


if __name__ == "__main__":
    main()
