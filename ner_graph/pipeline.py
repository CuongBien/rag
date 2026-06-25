import os
import json
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from openai import PermissionDeniedError, RateLimitError
import httpx # LlamaIndex Groq might throw httpx.HTTPStatusError on 429

from llama_index.core import PropertyGraphIndex, Settings, Document
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.property_graph import (
    DynamicLLMPathExtractor,
    ImplicitPathExtractor
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

def normalize_graph_entities(graph_store):
    """
    Thuật toán chuẩn hóa & gộp Entity tự động: 
    Tìm các Entity giống tên nhau (không phân biệt hoa thường, khoảng trắng)
    và dùng APOC để "zip" (gộp) chúng lại thành 1 siêu Node, chập toàn bộ quan hệ lại.
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
        config.neo4j_uri, config.neo4j_username, config.neo4j_password
    )
    
    # 1. Định nghĩa SCHEMA NGHIÊM NGẶT cho Não Trái
    kg_extractors = [
        DynamicLLMPathExtractor(
            llm=llm,
            max_triplets_per_chunk=10,
            num_workers=1, # Bẫy số 2: Giữ bằng 1 để không spam Groq API
            allowed_entity_types=["drug", "symptom", "enzyme", "mechanism", "disease", "protein"],
            allowed_relation_types=["interacts_with", "treats", "causes_side_effect", "metabolized_by", "inhibits", "induces"]
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
    
    # 2. Xử lý Checkpointing
    checkpoint_file = os.path.join(project_root, "processed_files.json")
    processed_files = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_files = json.load(f)
            
    documents = load_documents_from_data_dir(config.data_dir)
    print(f"[pipeline] Found {len(documents)} chunks total.")
    
    # Gom nhóm theo nguồn file (nếu có id) để quản lý
    # LlamaIndex Document thường có doc.metadata['file_name']
    docs_to_process = []
    for doc in documents:
        fname = doc.metadata.get('file_name', doc.id_)
        if fname not in processed_files:
            docs_to_process.append(doc)
            
    print(f"[pipeline] {len(docs_to_process)} chunks need to be processed.")
    
    # Vòng lặp chống Rate Limit
    for i, doc in enumerate(docs_to_process):
        fname = doc.metadata.get('file_name', doc.id_)
        print(f"[{i+1}/{len(docs_to_process)}] Extracting DDI for {fname}...")
        try:
            process_single_document(index, doc)
            # Lưu checkpoint thành công
            processed_files.append(fname)
            with open(checkpoint_file, "w") as f:
                json.dump(processed_files, f)
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
        config.neo4j_uri, config.neo4j_username, config.neo4j_password
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
