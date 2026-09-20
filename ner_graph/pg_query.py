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
   - Prefer this style: "Dua tren co so du lieu duoc hoc hien tai, khong tim thay thong tin ve ..."
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
    Hàm truy vấn chính xác (Não Trái logic) bằng Cypher:
    1. Tìm liên kết trực tiếp giữa 2 chất (kèm properties như severity, clinical_effect, dosage, reason).
    2. Nếu không có liên kết trực tiếp, tìm liên kết gián tiếp 1-hop qua Enzyme / Transporter / Target chung.
    """
    drug_a = drug_a.lower().strip()
    drug_b = drug_b.lower().strip()

    # 1. Truy vấn trực tiếp
    query_direct = f"""
    MATCH (d1)-[r]-(d2)
    WHERE (toLower(coalesce(d1.name, d1.id, '')) = '{drug_a}' OR toLower(coalesce(d1.id, d1.name, '')) = '{drug_a}')
      AND (toLower(coalesce(d2.name, d2.id, '')) = '{drug_b}' OR toLower(coalesce(d2.id, d2.name, '')) = '{drug_b}')
    RETURN type(r) as relation, properties(r) as props, coalesce(d1.name, d1.id) as s_name, coalesce(d2.name, d2.id) as t_name
    """
    try:
        results = graph_store.structured_query(query_direct)
        if results:
            return {"type": "direct", "records": results}
    except Exception as e:
        print(f"Cypher Direct Query Error: {e}")

    # 2. Truy vấn gián tiếp qua nút trung gian (Multi-hop Causal: e.g. Enzyme / Transporter)
    query_multihop = f"""
    MATCH (d1)-[r1]-(m)-[r2]-(d2)
    WHERE (toLower(coalesce(d1.name, d1.id, '')) = '{drug_a}' OR toLower(coalesce(d1.id, d1.name, '')) = '{drug_a}')
      AND (toLower(coalesce(d2.name, d2.id, '')) = '{drug_b}' OR toLower(coalesce(d2.id, d2.name, '')) = '{drug_b}')
      AND NOT (d1 = d2)
    RETURN coalesce(d1.name, d1.id) as s_name, type(r1) as r1_type, properties(r1) as r1_props,
           coalesce(m.name, m.id) as m_name, labels(m) as m_labels,
           type(r2) as r2_type, properties(r2) as r2_props,
           coalesce(d2.name, d2.id) as t_name
    LIMIT 5
    """
    try:
        results_hop = graph_store.structured_query(query_multihop)
        if results_hop:
            return {"type": "multihop", "records": results_hop}
    except Exception as e:
        print(f"Cypher Multi-hop Query Error: {e}")

    return {"type": "none", "records": []}

class AdaptiveGraphQueryEngine(CustomQueryEngine):
    """Định tuyến và Viết lại câu hỏi Graph dựa trên LLM 8B trọng tài."""
    base_engine: BaseQueryEngine
    referee_llm: object
    graph_store: object

    def custom_query(self, query_str: str):
        # 1. Trọng tài bóc tách thực thể thuốc/hóa chất/thực phẩm
        prompt = (
            f"Trích xuất tất cả các tên thuốc, hóa chất, thảo dược hoặc thực phẩm/đồ uống (như bưởi chùm, rượu, diazepam, v.v.) từ câu hỏi sau. "
            f"CHỈ trả về danh sách tên các chất, cách nhau bằng dấu phẩy (ví dụ: Diazepam, Lorazepam, Alcohol). "
            f"Nếu câu hỏi không chứa ít nhất 2 chất, CHỈ trả về chữ 'NONE'. "
            f"Không giải thích thêm. Câu hỏi: '{query_str}'"
        )
        try:
            decision = self.referee_llm.complete(prompt).text.strip()
        except Exception:
            decision = "NONE"

        print(f"\n[Adaptive Graph] Trong tai boc tach: {decision}")
        
        if decision.upper() != "NONE" and "," in decision:
            parts = [p.strip() for p in decision.split(",") if p.strip()]
            if len(parts) >= 2:
                import itertools
                
                nodes_dict = {}
                links = []
                hints = []
                found_any = False
                
                for a, b in itertools.combinations(parts, 2):
                    print(f"[Adaptive Graph] Kich hoat Cypher Exact Match cho: '{a}' va '{b}'")
                    res = exact_interaction_query(self.graph_store, a, b)
                    
                    if res["type"] == "direct":
                        found_any = True
                        records = res["records"]
                        a_key = a.lower()
                        b_key = b.lower()
                        
                        if a_key not in nodes_dict:
                            nodes_dict[a_key] = {"id": a_key, "name": a.capitalize(), "group": 1}
                        if b_key not in nodes_dict:
                            nodes_dict[b_key] = {"id": b_key, "name": b.capitalize(), "group": 2}

                        for r in records:
                            rel_name = r.get("relation", "INTERACTS_WITH")
                            props = r.get("props", {}) or {}
                            links.append({"source": a_key, "target": b_key, "label": rel_name})
                            
                            detail_items = []
                            for k in ["severity", "clinical_effect", "dosage", "reason", "role", "action"]:
                                if props.get(k):
                                    detail_items.append(f"{k}: {props[k]}")
                            detail_str = f" ({', '.join(detail_items)})" if detail_items else ""
                            hints.append(f"- Trực tiếp: {a.capitalize()} và {b.capitalize()} có liên kết '{rel_name}'{detail_str}")

                    elif res["type"] == "multihop":
                        found_any = True
                        records = res["records"]
                        for r in records:
                            s = r.get("s_name", a)
                            m = r.get("m_name", "Mediator")
                            t = r.get("t_name", b)
                            r1 = r.get("r1_type", "AFFECTS")
                            r2 = r.get("r2_type", "METABOLIZED_BY")
                            
                            s_key = s.lower()
                            m_key = m.lower()
                            t_key = t.lower()
                            
                            if s_key not in nodes_dict:
                                nodes_dict[s_key] = {"id": s_key, "name": s, "group": 1}
                            if m_key not in nodes_dict:
                                nodes_dict[m_key] = {"id": m_key, "name": m, "group": 3}
                            if t_key not in nodes_dict:
                                nodes_dict[t_key] = {"id": t_key, "name": t, "group": 2}
                                
                            links.append({"source": s_key, "target": m_key, "label": r1})
                            links.append({"source": t_key, "target": m_key, "label": r2})
                            hints.append(f"- Gián tiếp nhân quả: {s} -[{r1}]-> {m} <-[{r2}]- {t}")
                        
                if found_any:
                    graph_data = {
                        "nodes": list(nodes_dict.values()),
                        "links": links
                    }
                    hint_str = "\n".join(hints)
                    
                    answer_prompt = (
                        f"Bạn là chuyên gia Dược lý học lâm sàng. Dựa trên dữ liệu đồ thị tri thức y khoa đã được xác thực từ cơ sở dữ liệu sau đây:\n"
                        f"{hint_str}\n\n"
                        f"Hãy trả lời câu hỏi của người dùng một cách rõ ràng, chính xác, khách quan bằng tiếng Việt:\n"
                        f"Câu hỏi: '{query_str}'\n\n"
                        f"Yêu cầu:\n"
                        f"1. Khẳng định rõ các mối tương tác / cơ chế đã tìm thấy trong cơ sở dữ liệu đồ thị.\n"
                        f"2. Nếu là liên kết trực tiếp: giải thích rõ mức độ nghiêm trọng (severity), tác động lâm sàng hoặc khuyến cáo liều.\n"
                        f"3. Nếu là liên kết gián tiếp (qua enzyme/transporter/thụ thể): giải thích cơ chế dược động học (VD: ức chế men chuyển hóa làm tăng độc tính).\n"
                        f"4. Tuyệt đối KHÔNG nói 'không tìm thấy thông tin' đối với các thực thể đã được xác thực ở trên."
                    )
                    try:
                        llm_answer = self.referee_llm.complete(answer_prompt).text.strip()
                    except Exception as e:
                        llm_answer = f"Theo dữ liệu từ đồ thị tri thức y khoa:\n{hint_str}"
                    
                    metadata = {"graph_data": graph_data}
                    return Response(response=llm_answer, source_nodes=[], metadata=metadata)
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
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    if not drugs:
        return {"nodes": [], "links": []}
        
    driver = GraphDatabase.driver(uri, auth=(username, password))
    drugs_lower = [d.lower() for d in drugs]
    
    query = """
    MATCH (a)-[r]-(b)
    WHERE toLower(coalesce(a.id, a.name, '')) IN $drugs AND toLower(coalesce(b.id, b.name, '')) IN $drugs
    RETURN coalesce(a.id, a.name) AS a_id, type(r) AS rel_type, coalesce(b.id, b.name) AS b_id
    """
    
    nodes_dict = {}
    links = []
    
    for i, d in enumerate(drugs):
        nodes_dict[d.lower()] = {"id": d.lower(), "name": d.capitalize(), "group": i % 5 + 1}
        
    with driver.session(database=database) as session:
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
