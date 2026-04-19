"""Local embedding models (Neo4j stores vectors on Chunk + Entity nodes)."""

import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface.utils import (
    DEFAULT_EMBED_INSTRUCTION,
    DEFAULT_QUERY_BGE_INSTRUCTION_EN,
)


def resolve_embedding_device(requested: str | None) -> str:
    """
    Map EMBED_DEVICE to a device string safe for the installed PyTorch build.
    CPU-only wheels cannot use cuda even if .env says cuda.

    requested: value from EMBED_DEVICE, or None to auto-pick cuda if available else cpu.
    """
    if requested is not None and requested.strip() != "":
        raw = requested.strip()
        lower = raw.lower()
        if lower == "cuda" or lower.startswith("cuda:"):
            if not torch.cuda.is_available():
                print(
                    "[embed] EMBED_DEVICE requests CUDA but torch.cuda.is_available() is False; "
                    "using cpu. For GPU, install a CUDA-enabled PyTorch build from pytorch.org."
                )
                return "cpu"
            return raw
        return raw
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_bge_m3_embed_model(
    model_name: str,
    embed_batch_size: int,
    device: str | None,
    normalize_embeddings: bool,
    trust_remote_code: bool,
    show_progress_bar: bool,
) -> HuggingFaceEmbedding:
    """
    BGE-M3 via sentence-transformers; vectors written by Neo4jPropertyGraphStore
    when PropertyGraphIndex runs with embed_kg_nodes=True.

    model_name: Hugging Face hub id (e.g. BAAI/bge-m3).
    embed_batch_size: batch size for get_text_embedding_batch (memory vs throughput).
    device: torch device string or None (resolved via resolve_embedding_device).
    normalize_embeddings: L2-normalize vectors (recommended for cosine similarity).
    trust_remote_code: allow custom modeling code from the Hub (security tradeoff).
    show_progress_bar: show tqdm-style progress during batch embedding.
    """
    resolved_device = resolve_embedding_device(device)
    print(
        f"[embed] Loading BGE-M3 embedder model_name={model_name} "
        f"batch_size={embed_batch_size} device={resolved_device!r}"
    )
    return HuggingFaceEmbedding(
        model_name=model_name,
        max_length=None,  # use model default sequence length
        query_instruction=DEFAULT_QUERY_BGE_INSTRUCTION_EN,  # BGE query prefix for retrieval
        text_instruction=DEFAULT_EMBED_INSTRUCTION,  # BGE passage prefix for indexing
        normalize=normalize_embeddings,
        embed_batch_size=embed_batch_size,
        cache_folder=None,  # HF default cache dir
        trust_remote_code=trust_remote_code,
        device=resolved_device,
        callback_manager=None,  # no LlamaIndex callbacks
        parallel_process=False,  # single-process embedding
        target_devices=None,  # not used when parallel_process is False
        show_progress_bar=show_progress_bar,
    )
