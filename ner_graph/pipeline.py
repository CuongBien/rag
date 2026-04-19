import os

from openai import PermissionDeniedError

from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)

from .config import load_config
from .embeddings import create_bge_m3_embed_model
from .entity_merge import merge_similar_entities
from .graph_store import (
    create_graph_store,
    sanitize_entity_names,
    upsert_entity_lookup_ids,
)
from .ingest import load_documents_from_data_dir
from .llm_client import create_llm
from .pg_query import create_property_graph_query_engine
from .telemetry import setup_phoenix_tracing


def build_query_engine(project_root: str) -> BaseQueryEngine:
    # project_root: absolute path to repo root (used for data/ and .env resolution via cwd).
    print(f"[pipeline] Initializing query engine for project_root={project_root}")
    config = load_config(project_root)
    setup_phoenix_tracing()
    llm = create_llm(config.groq_model, config.groq_api_key, config.groq_api_base)
    Settings.llm = llm

    embed_model = create_bge_m3_embed_model(
        config.embed_model_name,
        config.embed_batch_size,
        config.embed_device,
        True,  # normalize_embeddings: L2-normalize for cosine similarity
        config.embed_trust_remote_code,
        True,  # show_progress_bar: tqdm during embedding batches
    )
    Settings.embed_model = embed_model

    graph_store = create_graph_store(
        config.neo4j_uri, config.neo4j_username, config.neo4j_password
    )
    sanitize_entity_names(graph_store)
    upsert_entity_lookup_ids(graph_store)

    documents = load_documents_from_data_dir(config.data_dir)
    print("[pipeline] Building PropertyGraphIndex...")
    try:
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
            show_progress=True,  # tqdm on extraction/embedding steps
            embed_kg_nodes=True,  # embed chunk + KG nodes; persist vectors in Neo4j
            kg_extractors=[
                SimpleLLMPathExtractor(
                    llm=llm,
                    num_workers=1,  # parallel LLM calls (Groq rate limits: keep low)
                    max_paths_per_chunk=10,  # max extracted relation paths per text chunk
                ),
                ImplicitPathExtractor(),  # adds MENTIONS etc. from chunk structure
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

    query_engine = create_property_graph_query_engine(
        index,
        True,  # include_text: feed chunk text into retriever context
        config.pg_path_depth,
        config.pg_vector_top_k,
        config.pg_rel_map_limit,
    )
    return query_engine


def answer_question(query_engine: BaseQueryEngine, question: str) -> str:
    print(f"[pipeline] Running question={question!r}")
    return str(query_engine.query(question))


def run_pipeline(project_root: str) -> None:
    query_engine = build_query_engine(project_root)
    question = "Ba Elena Rodriguez co moi lien he gian tiep nao voi ong Ly Hoang Nam khong?"
    print(f"\nQuestion: {question}")
    answer_text = answer_question(query_engine, question)
    # Keep Windows cp1252 terminals from crashing on Vietnamese Unicode output.
    safe_answer_text = answer_text.encode("ascii", errors="backslashreplace").decode(
        "ascii"
    )
    print(f"Answer: {safe_answer_text}")
