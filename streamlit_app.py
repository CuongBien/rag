import os

import streamlit as st
from llama_index.core.base.base_query_engine import BaseQueryEngine

from ner_graph.pipeline import answer_question, build_query_engine


def _get_query_engine(project_root: str) -> BaseQueryEngine:
    cached_engine = st.session_state.get("query_engine")
    if cached_engine is not None:
        return cached_engine
    st.session_state["query_engine"] = build_query_engine(project_root)
    return st.session_state["query_engine"]


def main() -> None:
    st.set_page_config(page_title="NER GraphRAG", page_icon="🔎", layout="wide")
    st.title("NER GraphRAG")
    st.caption("Vector -> Graph -> Text question answering over Neo4j")

    project_root = os.path.dirname(__file__)

    if st.button("Initialize pipeline", type="primary"):
        with st.spinner("Building query engine..."):
            _get_query_engine(project_root)
        st.success("Pipeline initialized.")

    question = st.text_input(
        "Question",
        value="Ba Elena Rodriguez co moi lien he gian tiep nao voi ong Ly Hoang Nam khong?",
    )

    if st.button("Ask"):
        if question.strip() == "":
            st.warning("Please enter a question.")
            return
        with st.spinner("Running retrieval and synthesis..."):
            query_engine = _get_query_engine(project_root)
            answer = answer_question(query_engine, question)
        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
