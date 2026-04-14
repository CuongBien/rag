import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    groq_model: str
    groq_api_base: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    merge_similarity_threshold: float
    merge_max_llm_checks: int
    data_dir: str


def _require_non_empty_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_config(project_root: str) -> AppConfig:
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
    )
