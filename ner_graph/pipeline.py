import os
import json
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from openai import PermissionDeniedError, RateLimitError
import httpx # LlamaIndex Groq might throw httpx.HTTPStatusError on 429

from llama_index.core import PropertyGraphIndex, Settings, Document
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.property_graph import (
    SchemaLLMPathExtractor,
    ImplicitPathExtractor,
)

from .config import load_config
from .embeddings import create_bge_m3_embed_model
from .graph_store import (
    create_graph_store,
    sanitize_entity_names,
    upsert_entity_lookup_ids,
)
from .ingest import load_documents_from_data_dir
from .llm_client import create_llm
from .pg_query import create_property_graph_query_engine, AdaptiveGraphQueryEngine
from .telemetry import setup_phoenix_tracing
from .schema_definition import (
    EntityTypes,
    RelationTypes,
    ENTITY_PROPERTIES,
    RELATION_PROPERTIES,
    VALIDATION_SCHEMA,
    MEDICAL_EXTRACTION_PROMPT,
    fetch_drug_smiles,
)

# Hàm xử lý lỗi 429 Rate Limit bằng Tenacity
# Tự động thử lại tối đa 5 lần, đợi lũy thừa (2s, 4s, 8s...)
@retry(
    wait=wait_exponential(multiplier=2, min=4, max=30),
    stop=stop_after_attempt(5),
    retry=(retry_if_exception_type(Exception)) # Catch all network/rate limits temporarily
)
def process_single_document(index: PropertyGraphIndex, doc: Document):
    """Xử lý từng document một để chống văng script"""
    index.insert(doc)

def migrate_legacy_schema(graph_store):
    """
    Chuẩn hóa các nhãn và quan hệ cũ từ dạng tự do sang 9 Thực thể & 9 Quan hệ chuẩn UPPERCASE.
    """
    print("[pipeline] Migrating legacy Neo4j labels to standard schema...")
    label_mappings = [
        ("drug", "DRUG"),
        ("disease", "DISEASE"),
        ("symptom", "SIDE_EFFECT"),
        ("enzyme", "ENZYME"),
        ("protein", "TARGET"),
        ("mechanism", "TARGET"),
        ("condition", "CONDITION"),
    ]
    for old_label, new_label in label_mappings:
        try:
            graph_store.structured_query(
                f"MATCH (n:`{old_label}`) SET n:`{new_label}` REMOVE n:`{old_label}`"
            )
        except Exception:
            pass

    rel_mappings = [
        ("interacts_with", "INTERACTS_WITH"),
        ("treats", "TREATS"),
        ("causes_side_effect", "CAUSES_SIDE_EFFECT"),
        ("metabolized_by", "METABOLIZED_BY"),
    ]
    for old_rel, new_rel in rel_mappings:
        try:
            graph_store.structured_query(
                f"MATCH (a)-[r:`{old_rel}`]->(b) MERGE (a)-[r2:`{new_rel}`]->(b) SET r2 += properties(r) DELETE r"
            )
        except Exception:
            pass

def enrich_drug_smiles(graph_store):
    """
    Tra cứu và bổ sung thuộc tính SMILES và name_lower cho tất cả các node DRUG trong đồ thị.
    """
    print("[pipeline] Enriching DRUG nodes with SMILES from NIH Cactus / PubChem...")
    try:
        drugs = graph_store.structured_query(
            "MATCH (d:DRUG) WHERE d.name IS NOT NULL AND (d.smiles IS NULL OR d.smiles = '') RETURN d.name AS name LIMIT 100"
        )
        count = 0
        for row in drugs:
            name = row.get("name")
            if not name:
                continue
            smiles = fetch_drug_smiles(name)
            if smiles:
                graph_store.structured_query(
                    "MATCH (d:DRUG {name: $name}) SET d.smiles = $smiles, d.name_lower = toLower($name)",
                    param_map={"name": name, "smiles": smiles}
                )
                count += 1
            else:
                graph_store.structured_query(
                    "MATCH (d:DRUG {name: $name}) SET d.name_lower = toLower($name)",
                    param_map={"name": name}
                )
        print(f"[pipeline] SMILES enrichment completed for {count} drugs.")
    except Exception as e:
        print(f"[pipeline] Warning during SMILES enrichment: {e}")

def normalize_graph_entities(graph_store):
    """
    Thuật toán chuẩn hóa & gộp Entity tự động: 
    1. APOC Deduplication gộp node trùng tên
    2. Migrate nhãn/quan hệ cũ sang chuẩn mới
    3. Làm giàu SMILES từ NIH Cactus / PubChem
    """
    print("[pipeline] Running APOC Entity Resolution (Deduplication)...")
    try:
        query = """
        MATCH (e:__Entity__)
        WHERE e.name IS NOT NULL
        WITH toLower(trim(e.name)) AS cleanName, collect(e) AS duplicates
        WHERE size(duplicates) > 1
        CALL apoc.refactor.mergeNodes(duplicates, {
          properties: "overwrite", 
          mergeRels: true
        })
        YIELD node
        RETURN count(node) AS merged_count
        """
        results = graph_store.structured_query(query)
        if results and len(results) > 0:
            print(f"[pipeline] Deduplication completed. Merged {results[0].get('merged_count', 0)} duplicate clusters into super nodes.")
        else:
            print("[pipeline] Deduplication completed. No duplicates found.")
    except Exception as e:
        print(f"[pipeline] Could not run APOC deduplication (is APOC installed?): {e}")

    # Chạy di chuyển nhãn cũ & làm giàu SMILES
    migrate_legacy_schema(graph_store)
    enrich_drug_smiles(graph_store)

def build_knowledge_graph(project_root: str):
    """
    Hàm mới: Xây dựng Knowledge Graph có Checkpoint (Bẫy số 2)
    """
    print(f"[pipeline] Building Schema-driven Knowledge Graph for project_root={project_root}")
    config = load_config(project_root)
    setup_phoenix_tracing()
    
    llm = create_llm(config)
    Settings.llm = llm

    embed_model = create_bge_m3_embed_model(
        config.embed_model_name,
        config.embed_batch_size,
        config.embed_device,
        True,  # normalize_embeddings: L2-normalize for cosine similarity
        config.embed_trust_remote_code,
        True,  # show_progress_bar
    )
    Settings.embed_model = embed_model

    graph_store = create_graph_store(
        config.neo4j_uri,
        config.neo4j_username,
        config.neo4j_password,
        config.neo4j_database,
    )
    
    # 1. Định nghĩa SCHEMA CHUẨN Y DƯỢC (9 Thực Thể & 9 Mối Quan Hệ)
    kg_extractors = [
        SchemaLLMPathExtractor(
            llm=llm,
            possible_entities=EntityTypes,
            possible_entity_props=ENTITY_PROPERTIES,
            possible_relations=RelationTypes,
            possible_relation_props=RELATION_PROPERTIES,
            kg_validation_schema=VALIDATION_SCHEMA,
            extract_prompt=MEDICAL_EXTRACTION_PROMPT,
            strict=True,
            num_workers=1,
            max_triplets_per_chunk=12,
        ),
        ImplicitPathExtractor(),
    ]
    
    # Khởi tạo Index nối với Neo4j
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        kg_extractors=kg_extractors,
        embed_model=embed_model,
        embed_kg_nodes=True,
    )
    
    # 2. Xử lý Checkpointing (Tự động quét cả file JSON lẫn DB Neo4j)
    checkpoint_file = os.path.join(project_root, "processed_files.json")
    processed_set = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                processed_set.update(json.load(f))
        except Exception as e:
            print(f"[pipeline] Warning reading checkpoint file: {e}")

    # Đồng bộ tự động các file đã có sẵn trong Neo4j để tuyệt đối không chạy lại
    try:
        db_records = graph_store.structured_query(
            "MATCH (n) WHERE n.file_name IS NOT NULL RETURN DISTINCT n.file_name AS fn"
        )
        for row in db_records:
            if row.get("fn"):
                processed_set.add(row["fn"])
    except Exception as e:
        print(f"[pipeline] Warning checking existing files in Neo4j: {e}")

    # Cập nhật lại file checkpoint
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(sorted(list(processed_set)), f, indent=2)

    documents = load_documents_from_data_dir(config.data_dir)
    print(f"[pipeline] Found {len(documents)} chunks total.")
    
    # Gom nhóm theo nguồn file (nếu có id) để quản lý
    docs_to_process = []
    for doc in documents:
        fname = doc.metadata.get('file_name', doc.id_)
        if fname not in processed_set:
            docs_to_process.append(doc)
        else:
            print(f"[pipeline] ⏭️ Đã có trong Neo4j: Bỏ qua {fname}")
            
    print(f"\n[pipeline] 🎯 {len(docs_to_process)} chunks còn lại cần trích xuất (Đã bỏ qua {len(processed_set)} files đã làm).")
    
    # Vòng lặp chống Rate Limit
    for i, doc in enumerate(docs_to_process):
        fname = doc.metadata.get('file_name', doc.id_)
        print(f"[{i+1}/{len(docs_to_process)}] Extracting DDI for {fname}...")
        try:
            process_single_document(index, doc)
            # Lưu checkpoint thành công
            processed_set.add(fname)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(sorted(list(processed_set)), f, indent=2)
            time.sleep(1) # Nghỉ xả hơi nhẹ giữa các chunks
        except Exception as e:
            print(f"[pipeline] ❌ Failed completely on {fname} after retries.")
            print(f"Chi tiết lỗi: {repr(e)}")
            import traceback
            traceback.print_exc()
            break # Dừng script an toàn
            
    # Chạy Normalize toàn bộ Graph (Bẫy số 1)
    normalize_graph_entities(graph_store)
    
    print("[pipeline] Knowledge Graph build completed!")
    return index

def build_query_engine(project_root: str) -> BaseQueryEngine:
    config = load_config(project_root)
    setup_phoenix_tracing()
    
    llm = create_llm(config)
    Settings.llm = llm

    embed_model = create_bge_m3_embed_model(
        config.embed_model_name,
        config.embed_batch_size,
        config.embed_device,
        True,
        config.embed_trust_remote_code,
        False,
    )
    Settings.embed_model = embed_model
    graph_store = create_graph_store(
        config.neo4j_uri,
        config.neo4j_username,
        config.neo4j_password,
        config.neo4j_database,
    )
    
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=embed_model,
    )
    
    query_engine = create_property_graph_query_engine(
        index,
        True,  # include_text
        config.pg_path_depth,
        config.pg_vector_top_k,
        config.pg_rel_map_limit,
    )
    
    adaptive_engine = AdaptiveGraphQueryEngine(
        base_engine=query_engine,
        referee_llm=llm,
        graph_store=graph_store
    )
    return adaptive_engine

def answer_question(query_engine: BaseQueryEngine, question: str) -> str:
    return str(query_engine.query(question))


def run_pipeline(project_root: str) -> None:
    """Xây dựng Knowledge Graph từ dữ liệu và chạy câu hỏi truy vấn mẫu."""
    print(f"[pipeline] Running Graph RAG pipeline for project_root={project_root}...")
    build_knowledge_graph(project_root)
    query_engine = build_query_engine(project_root)
    question = "Paracetamol có tương tác nguy hiểm nào với Warfarin không?"
    print(f"\n[pipeline] Question: {question}")
    answer_text = answer_question(query_engine, question)
    print(f"\n[pipeline] Answer:\n{answer_text}")

