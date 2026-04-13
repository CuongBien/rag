import os
from typing import Final

from dotenv import load_dotenv

from llama_index.core import Document, Settings
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLMMetadata
from llama_index.llms.openai.utils import openai_modelname_to_contextsize

class GroqOpenAI(OpenAI):
    """OpenAI-compatible wrapper for Groq that bypasses the model allowlist."""

    @property
    def metadata(self) -> LLMMetadata:
        # Groq's llama-3.3-70b-versatile has a 128k context window
        return LLMMetadata(
            context_window=128_000,
            num_output=self.max_tokens or 2048,
            is_chat_model=True,
            model_name=self.model,
        )

# Groq exposes an OpenAI-compatible API; llama-index-llms-groq conflicts with llama-index-llms-openai 0.7.x.
GROQ_OPENAI_BASE_URL: Final[str] = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL: Final[str] = "llama-3.3-70b-versatile"


def _require_non_empty_env(name: str) -> str:
    value: str | None = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Add it to a .env file in the project root or export it in your shell."
        )
    return value


def main() -> None:
    load_dotenv()

    groq_api_key: str = _require_non_empty_env("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    print(f"[debug] Groq OpenAI-compatible endpoint, model={groq_model}")

    llm = GroqOpenAI(
        model=groq_model,
        temperature=0.0,
        max_retries=3,
        timeout=120.0,
        reuse_client=True,
        api_key=groq_api_key,
        api_base=GROQ_OPENAI_BASE_URL,
    )
    Settings.llm = llm

    # Groq has no embeddings service: disable KG node embeddings so Settings.embed_model is not required.
    embed_kg_nodes: bool = False

    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = _require_non_empty_env("NEO4J_PASSWORD")

    print("[debug] Connecting to Neo4j...")
    graph_store = Neo4jPropertyGraphStore(
        username=neo4j_username,
        password=neo4j_password,
        url=neo4j_uri,
    )

    text_data = """
Tập đoàn công nghệ TechCorp được thành lập vào năm 2010 bởi John Doe. 
Trụ sở chính của TechCorp nằm tại San Francisco, Mỹ. 
Năm 2022, TechCorp đã mua lại startup AI có tên là SmartBrain với giá 1 tỷ USD. 
Alice Smith, một kỹ sư dữ liệu xuất sắc, trước đây làm việc cho SmartBrain, 
hiện đã trở thành Giám đốc AI của TechCorp sau vụ mua lại này.
"""
    documents = [Document(text=text_data)]

    print(
        "[debug] PropertyGraphIndex.from_documents(embed_kg_nodes=False) — graph via LLM only."
    )
    print("[debug] Calling Groq for triplet extraction...")
    index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        show_progress=True,
        embed_kg_nodes=embed_kg_nodes,
    )
    print("[debug] Upsert to Neo4j completed.")

    query_engine = index.as_query_engine(include_text=True)

    question = "Alice Smith đang làm việc ở đâu và công ty đó do ai thành lập?"
    print(f"\nCâu hỏi: {question}")
    response = query_engine.query(question)
    print(f"Trả lời: {response}")


if __name__ == "__main__":
    main()
