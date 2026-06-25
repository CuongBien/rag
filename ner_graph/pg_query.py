"""Property graph RAG: BGE vectors on __Entity__ + multi-hop subgraph (get_rel_map)."""

from llama_index.core import PropertyGraphIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.base.response.schema import Response


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

def exact_interaction_query(graph_store, drug_a: str, drug_b: str):
    """
    Hàm truy vấn chính xác (Não Trái logic) bằng Cypher
    Không dùng vector similarity, chỉ check cấu trúc rành mạch
    """
    drug_a = drug_a.lower().strip()
    drug_b = drug_b.lower().strip()
    query = f"""
    MATCH (d1 {{name: '{drug_a}'}})-[r]-(d2 {{name: '{drug_b}'}})
    RETURN type(r) as relation
    """
    try:
        results = graph_store.structured_query(query)
        if results:
            return [r['relation'] for r in results]
        return []
    except Exception as e:
        print(f"Cypher Error: {e}")
        return []

class AdaptiveGraphQueryEngine(CustomQueryEngine):
    """Định tuyến và Viết lại câu hỏi Graph dựa trên LLM 8B trọng tài."""
    base_engine: BaseQueryEngine
    referee_llm: object
    graph_store: object

    def custom_query(self, query_str: str):
        # 1. Trọng tài 8B bóc tách thực thể
        prompt = (
            f"Trích xuất chính xác 2 tên thuốc/hóa chất từ câu hỏi sau. "
            f"CHỈ trả về 2 tên thuốc, cách nhau bằng dấu phẩy. "
            f"Nếu câu hỏi không có đủ 2 tên thuốc, CHỈ trả về chữ 'NONE'. "
            f"Không giải thích thêm. Câu hỏi: '{query_str}'"
        )
        try:
            decision = self.referee_llm.complete(prompt).text.strip()
        except Exception:
            decision = "NONE"

        print(f"\n[Adaptive Graph] Trong tai 8b boc tach: {decision}")
        
        if decision.upper() != "NONE" and "," in decision:
            parts = [p.strip() for p in decision.split(",")]
            if len(parts) >= 2:
                import itertools
                
                nodes_dict = {}
                links = []
                found_any = False
                
                for a, b in itertools.combinations(parts, 2):
                    print(f"[Adaptive Graph] Kich hoat Cypher Exact Match cho: '{a}' va '{b}'")
                    relations = exact_interaction_query(self.graph_store, a, b)
                    if relations:
                        found_any = True
                        relations = list(set(relations))
                        rel_str = ", ".join(relations)
                        print(f"[Adaptive Graph] Tim thay bang Cypher: {rel_str}")
                        
                        if a not in nodes_dict:
                            nodes_dict[a] = {"id": a, "name": a.capitalize(), "group": len(nodes_dict) % 5 + 1}
                        if b not in nodes_dict:
                            nodes_dict[b] = {"id": b, "name": b.capitalize(), "group": len(nodes_dict) % 5 + 1}
                            
                        links.append({"source": a, "target": b, "label": rel_str})
                        
                if found_any:
                    graph_data = {
                        "nodes": list(nodes_dict.values()),
                        "links": links
                    }
                    
                    hints = []
                    for link in links:
                        hints.append(f"{link['source']} và {link['target']} có quan hệ '{link['label']}'")
                    hint_str = "; ".join(hints)
                    
                    rich_query = f"{query_str}\n(Gợi ý từ DB: {hint_str}). Trả lời tự nhiên bằng tiếng Việt."
                    res = self.base_engine.query(rich_query)
                    
                    metadata = res.metadata or {}
                    metadata["graph_data"] = graph_data
                    return Response(response=res.response, source_nodes=res.source_nodes, metadata=metadata)
                else:
                    print(f"[Adaptive Graph] Cypher khong tim thay. Chuyen sang Graph Walk mo...")
        
        # Nếu không phải từ 2 thuốc trở lên, hoặc Cypher thất bại -> Fallback về Base Graph Engine
        print(f"[Adaptive Graph] Fallback: Chay thuat toan Graph Walk goc...")
        return self.base_engine.query(query_str)

def get_sandbox_graph_data(drugs: list[str]) -> dict:
    import os
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    if not drugs:
        return {"nodes": [], "links": []}
        
    driver = GraphDatabase.driver(uri, auth=(username, password))
    drugs_lower = [d.lower() for d in drugs]
    
    query = """
    MATCH (a)-[r]-(b)
    WHERE toLower(a.id) IN $drugs AND toLower(b.id) IN $drugs
    RETURN a.id AS a_id, type(r) AS rel_type, b.id AS b_id
    """
    
    nodes_dict = {}
    links = []
    
    for i, d in enumerate(drugs):
        nodes_dict[d.lower()] = {"id": d.lower(), "name": d.capitalize(), "group": i % 5 + 1}
        
    with driver.session() as session:
        result = session.run(query, drugs=drugs_lower)
        
        grouped_links = {}
        for record in result:
            a_id = record["a_id"].lower()
            b_id = record["b_id"].lower()
            rel = record["rel_type"]
            
            # Sort source/target so curvature math is consistent directionally
            pair = tuple(sorted([a_id, b_id]))
            if pair not in grouped_links:
                grouped_links[pair] = []
            
            # Force same direction for identical relations (A->B and B->A become just A->B)
            link_obj = {"source": pair[0], "target": pair[1], "label": rel}
            if link_obj not in grouped_links[pair]:
                grouped_links[pair].append(link_obj)
            
        for pair, rels in grouped_links.items():
            count = len(rels)
            for i, link_obj in enumerate(rels):
                if count == 1:
                    curvature = 0.0
                else:
                    curvature = 0.5 / (count - 1) * i - 0.25
                
                link_obj["curvature"] = curvature
                links.append(link_obj)
                
    return {"nodes": list(nodes_dict.values()), "links": links}
