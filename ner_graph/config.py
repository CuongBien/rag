import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str  # Secret for Groq chat-completions (OpenAI-compatible API).
    groq_model: str  # Model id on Groq (e.g. llama-3.3-70b-versatile).
    groq_api_base: str  # Base URL for the OpenAI-compatible client (Groq endpoint).
    neo4j_uri: str  # Bolt URI for Neo4j (host/port).
    neo4j_username: str  # Neo4j database user.
    neo4j_password: str  # Neo4j database password.
    merge_similarity_threshold: float  # Min fuzzy score (0–1) to consider entity pairs for LLM merge.
    merge_max_llm_checks: int  # Cap on LLM same-entity checks per run (cost/latency bound).
    data_dir: str  # Directory scanned for **/*.pdf (absolute path under project).
    embed_model_name: str  # Hugging Face model id for BGE/sentence-transformers embeddings.
    embed_batch_size: int  # Texts per embedding batch (VRAM/RAM tradeoff).
    embed_device: str | None  # Torch device hint (e.g. cuda); None lets code auto-pick.
    embed_trust_remote_code: bool  # Whether to allow remote code on HF Hub for the embed model.
    pg_path_depth: int  # Multi-hop depth for graph context (Neo4j variable-length paths, r*1..depth).
    pg_vector_top_k: int  # Top-k similar __Entity__ nodes from vector search before expanding graph.
    pg_rel_map_limit: int  # Max relationship rows collected when expanding subgraph around seeds.


def _require_non_empty_env(name: str) -> str:
    # name: environment variable key that must be present and non-blank.
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_config(project_root: str) -> AppConfig:
    # project_root: repository root; used to resolve data_dir = project_root/data.
    load_dotenv()
    print("[config] Loading environment variables...")

    groq_api_key = _require_non_empty_env("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_api_base = "https://api.groq.com/openai/v1"
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = _require_non_empty_env("NEO4J_PASSWORD")
    merge_similarity_threshold = float(os.getenv("MERGE_SIMILARITY_THRESHOLD", "0.6"))
    merge_max_llm_checks = int(os.getenv("MERGE_MAX_LLM_CHECKS", "50"))
    data_dir = os.path.join(project_root, "data")
    embed_model_name = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
    embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "8"))
    embed_device_raw = os.getenv("EMBED_DEVICE")
    embed_device = None if embed_device_raw is None or embed_device_raw.strip() == "" else embed_device_raw.strip()
    embed_trust_remote_code = os.getenv("EMBED_TRUST_REMOTE_CODE", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    pg_path_depth = int(os.getenv("PG_PATH_DEPTH", "3"))
    pg_vector_top_k = int(os.getenv("PG_VECTOR_TOP_K", "8"))
    pg_rel_map_limit = int(os.getenv("PG_REL_MAP_LIMIT", "40"))
    if pg_path_depth < 1:
        raise RuntimeError("PG_PATH_DEPTH must be >= 1")
    if pg_vector_top_k < 1:
        raise RuntimeError("PG_VECTOR_TOP_K must be >= 1")
    if pg_rel_map_limit < 1:
        raise RuntimeError("PG_REL_MAP_LIMIT must be >= 1")

    return AppConfig(
        groq_api_key=groq_api_key,
        groq_model=groq_model,
        groq_api_base=groq_api_base,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        merge_similarity_threshold=merge_similarity_threshold,
        merge_max_llm_checks=merge_max_llm_checks,
        data_dir=data_dir,
        embed_model_name=embed_model_name,
        embed_batch_size=embed_batch_size,
        embed_device=embed_device,
        embed_trust_remote_code=embed_trust_remote_code,
        pg_path_depth=pg_path_depth,
        pg_vector_top_k=pg_vector_top_k,
        pg_rel_map_limit=pg_rel_map_limit,
    )
