import os

from openai import PermissionDeniedError

from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)

from .config import load_config
from .entity_merge import merge_similar_entities
from .graph_store import create_graph_store, sanitize_entity_names
from .ingest import load_documents_from_data_dir
from .llm_client import create_llm


def run_pipeline(project_root: str) -> None:
    config = load_config(project_root)
    llm = create_llm(config.groq_model, config.groq_api_key, config.groq_api_base)
    Settings.llm = llm

    graph_store = create_graph_store(
        config.neo4j_uri, config.neo4j_username, config.neo4j_password
    )
    sanitize_entity_names(graph_store)

    documents = load_documents_from_data_dir(config.data_dir)
    print("[pipeline] Building PropertyGraphIndex...")
    try:
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
            show_progress=True,
            embed_kg_nodes=False,
            kg_extractors=[
                SimpleLLMPathExtractor(
                    llm=llm,
                    num_workers=1,
                    max_paths_per_chunk=10,
                ),
                ImplicitPathExtractor(),
            ],
        )
    except PermissionDeniedError as exc:
        raise RuntimeError("Groq API denied (HTTP 403). Check GROQ_API_KEY.") from exc

    merged_count = merge_similar_entities(
        graph_store,
        llm,
        config.merge_similarity_threshold,
        config.merge_max_llm_checks,
    )
    print(f"[pipeline] Entity merge completed, merged_count={merged_count}")

    query_engine = index.as_query_engine(include_text=True)
    question = "Ong Ly Hoang Nam co moi quan he gian tiep nao voi ba Elena Rodriguez?"
    print(f"\nQuestion: {question}")
    print(f"Answer: {query_engine.query(question)}")
