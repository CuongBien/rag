"""Property graph RAG: BGE vectors on __Entity__ + multi-hop subgraph (get_rel_map)."""

from llama_index.core import PropertyGraphIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import ResponseMode


GRAPH_GROUNDED_QA_TEMPLATE = PromptTemplate(
    """
You are a GraphRAG analyst. Use ONLY the provided graph context.

Rules:
1) If context contains relevant facts, describe the relationships in detail (direct or indirect).
2) Treat a valid indirect chain as a valid relationship answer.
3) If context does NOT contain any relevant entity or relationship for the question:
   - Reply politely that the current material does not mention this topic.
   - DO NOT use technical terms such as "node", "do thi", "duong noi", "database".
   - Prefer this style: "Dua tren bao cao chien luoc nam 2024, hien khong co thong tin ve ..."
4) Do NOT output "khong co thong tin truc tiep" when an indirect relationship exists.
5) Always keep the answer objective, precise, and in natural Vietnamese for end users.
6) Keep the response concise (2-4 sentences), with clear evidence phrasing when available.

Context:
{context_str}

Question:
{query_str}

Answer:
"""
)


def create_property_graph_query_engine(
    index: PropertyGraphIndex,
    include_text: bool,
    path_depth: int,
    similarity_top_k: int,
    rel_map_limit: int,
) -> BaseQueryEngine:
    """
    Wrap PropertyGraphIndex.as_query_engine with VectorContextRetriever multi-hop.

    Neo4j vector search uses the stored entity embedding index; path_depth expands
    triplets along r*1..depth (MENTIONS edges excluded by the graph store).

    index: built PropertyGraphIndex (Neo4j store + optional vectors).
    include_text: attach source chunk text to retrieved graph context when available.
    path_depth: hops for subgraph expansion around vector-hit entities (get_rel_map depth).
    similarity_top_k: how many nearest __Entity__ nodes to take from vector search.
    rel_map_limit: cap on rows returned when walking related triplets from seeds.
    """
    print(
        "[query] PropertyGraphQueryEngine "
        f"path_depth={path_depth} similarity_top_k={similarity_top_k} "
        f"rel_map_limit={rel_map_limit} include_text={include_text}"
    )
    return index.as_query_engine(
        include_text=include_text,
        path_depth=path_depth,  # passed to VectorContextRetriever + LLMSynonymRetriever
        similarity_top_k=similarity_top_k,  # VectorContextRetriever: entity ANN top-k
        limit=rel_map_limit,  # Neo4j get_rel_map row cap per retriever call
        response_mode=ResponseMode.TREE_SUMMARIZE,  # aggregate multi-hop evidence as a tree
        text_qa_template=GRAPH_GROUNDED_QA_TEMPLATE,
        summary_template=GRAPH_GROUNDED_QA_TEMPLATE,
    )
