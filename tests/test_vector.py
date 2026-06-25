import os
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from llama_index.core import Settings
from ner_graph.vector_store import build_vector_index
from ner_graph.llm_client import create_llm
from ner_graph.config import load_config

def main():
    print("🚀 BẮT ĐẦU NẠP DỮ LIỆU VÀO NÃO PHẢI (CHROMADB)...")
    print("-" * 50)
    
    # 1. Load config để lấy thông tin model Groq
    config = load_config(PROJECT_ROOT)
    
    # 2. Ưu tiên lấy GROQ_API_KEY từ biến môi trường (như bạn yêu cầu)
    groq_api_key = os.environ.get("GROQ_API_KEY", config.groq_api_key)
    
    if not groq_api_key:
        print("❌ LỖI: Chưa có GROQ_API_KEY. Bạn hãy set biến môi trường GROQ_API_KEY hoặc điền vào file .env rồi chạy lại nhé!")
        sys.exit(1)

    # 3. Khởi tạo LLM qua Groq và gán cho LlamaIndex (thay thế OpenAI)
    llm = create_llm(config.groq_model, groq_api_key, config.groq_api_base)
    Settings.llm = llm
    
    # Bước này sẽ băm nhỏ text và dùng model BGE-M3 để lưu vào chroma_db
    index = build_vector_index(PROJECT_ROOT)
    
    print("\n" + "=" * 50)
    print("🔍 THỬ NGHIỆM TRUY VẤN TỪ NÃO PHẢI")
    print("Câu hỏi: What is the medical use of Paracetamol?")
    print("-" * 50)
    
    # Tạo query engine (đã tích hợp sẵn Groq LLM ở trên)
    query_engine = index.as_query_engine(similarity_top_k=2)
    response = query_engine.query("What is the medical use of Paracetamol?")
    
    print("🤖 Câu trả lời:\n")
    print(response)
    print("=" * 50)

if __name__ == "__main__":
    main()
