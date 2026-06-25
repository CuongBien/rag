import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from .embeddings import create_bge_m3_embed_model
from .config import load_config

def build_vector_index(project_root: str) -> VectorStoreIndex:
    print(f"[vector_store] Building Vector Index for project_root={project_root}")
    config = load_config(project_root)
    
    # Define paths
    data_dir = os.path.join(project_root, "data_vector")
    db_dir = os.path.join(project_root, "chroma_db")
    
    if not os.path.exists(data_dir):
        print(f"[vector_store] Warning: {data_dir} does not exist. Please run download_wiki.py first.")
        os.makedirs(data_dir, exist_ok=True)
    
    # Initialize ChromaDB client
    db = chromadb.PersistentClient(path=db_dir)
    chroma_collection = db.get_or_create_collection("wikipedia_compounds")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Initialize embedding model using existing config
    embed_model = create_bge_m3_embed_model(
        config.embed_model_name,
        config.embed_batch_size,
        config.embed_device,
        True,
        config.embed_trust_remote_code,
        True
    )
    
    # Kiểm tra xem ChromaDB đã có dữ liệu chưa
    if chroma_collection.count() > 0:
        print(f"[vector_store] Found {chroma_collection.count()} embeddings in ChromaDB. Loading existing index...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
    else:
        # Load documents
        print(f"[vector_store] Loading documents from {data_dir}...")
        try:
            documents = SimpleDirectoryReader(data_dir).load_data()
            print(f"[vector_store] Loaded {len(documents)} document chunks.")
        except ValueError:
            print("[vector_store] No files found in data_vector. Returning empty index.")
            documents = []
        
        # Create index
        print("[vector_store] Indexing documents into ChromaDB...")
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            embed_model=embed_model,
            show_progress=True
        )
        
    print("[vector_store] Vector Index is ready.")
    return index

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

class AdaptiveVectorQueryEngine(CustomQueryEngine):
    """Định tuyến câu hỏi Vector dựa trên LLM 8B trọng tài."""
    base_engine: object
    hyde_engine: object
    referee_llm: object

    def custom_query(self, query_str: str):
        # 1. Trọng tài 8B phán xử câu hỏi
        prompt = (
            f"Phân tích câu hỏi: '{query_str}'.\n"
            "1. Nếu câu hỏi dưới 12 chữ, hoặc hỏi về sự kiện lịch sử (năm nào, ai tìm ra, ở đâu), hãy trả lời: HYDE\n"
            "2. Nếu câu hỏi lủng củng, đa ngôn ngữ, hãy trả lời: REWRITE\n"
            "3. Chỉ trả lời DIRECT nếu câu hỏi là một đoạn văn dài mô tả chi tiết.\n"
            "CHỈ ĐƯỢC PHÉP TRẢ LỜI ĐÚNG 1 TỪ (HYDE, REWRITE, hoặc DIRECT). KHÔNG GIẢI THÍCH."
        )
        try:
            decision = self.referee_llm.complete(prompt).text.strip().upper()
        except Exception:
            decision = "DIRECT"

        if "HYDE" in decision:
            print(f"\n[Adaptive Vector] ⚖️ Trọng tài 8b quyết định: HYDE (Sinh ngữ cảnh giả)")
            return self.hyde_engine.query(query_str)
        elif "REWRITE" in decision:
            print(f"\n[Adaptive Vector] ⚖️ Trọng tài 8b quyết định: REWRITE (Viết lại từ khóa)")
            rewrite_prompt = (
                f"Hãy viết lại câu hỏi sau sang tiếng Anh, sử dụng các từ khóa chuyên ngành Y/Hóa học "
                f"(như synthesize, interact, mechanism) để tìm kiếm tài liệu chuẩn xác nhất. "
                f"CHỈ TRẢ LỜI CÂU HỎI MỚI, KHÔNG GIẢI THÍCH: '{query_str}'"
            )
            try:
                new_query = self.referee_llm.complete(rewrite_prompt).text.strip()
                new_query = new_query.replace('"', '').replace("'", "")
                print(f"[Adaptive Vector] 🔄 Đã bẻ lái câu hỏi thành: {new_query}")
                return self.base_engine.query(new_query)
            except Exception:
                return self.base_engine.query(query_str)
        else:
            print(f"\n[Adaptive Vector] ⚖️ Trọng tài 8b quyết định: DIRECT (Tìm kiếm trực tiếp)")
            return self.base_engine.query(query_str)

def get_vector_query_engine(project_root: str, llm=None):
    from ner_graph.llm_client import GroqOpenAI
    config = load_config(project_root)
    
    index = build_vector_index(project_root)
    base_engine = index.as_query_engine(similarity_top_k=4, llm=llm)
    
    # Khởi tạo Trọng tài 8B riêng biệt siêu nhẹ
    import os
    groq_api_key = os.environ.get("GROQ_API_KEY", config.groq_api_key)
    referee_llm = GroqOpenAI(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        api_base=config.groq_api_base,
        temperature=0.0
    )
    
    # Engine HyDE sinh văn bản nháp
    hyde_transform = HyDEQueryTransform(llm=referee_llm)
    hyde_engine = TransformQueryEngine(base_engine, query_transform=hyde_transform)
    
    # Đóng gói vào Adaptive Engine
    return AdaptiveVectorQueryEngine(
        base_engine=base_engine,
        hyde_engine=hyde_engine,
        referee_llm=referee_llm
    )
