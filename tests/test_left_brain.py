import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Thêm thư mục gốc vào sys.path để import được ner_graph
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from ner_graph.pipeline import build_knowledge_graph
from ner_graph.pg_query import exact_interaction_query
from ner_graph.config import load_config
from ner_graph.graph_store import create_graph_store

def main():
    import os
    # Nếu có GEMINI_API_KEY thì xóa để bắt buộc dùng GROQ
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
        
    # Tắt thông báo rác của Telemetry
    os.environ["PHOENIX_ENABLED"] = "0"
    # Ép dùng CPU cho Embeddings thay vì CUDA để khỏi tranh giành VRAM GPU với Streamlit
    os.environ["EMBED_DEVICE"] = "cpu"

    print("🧠 BẮT ĐẦU XÂY DỰNG NÃO TRÁI (GRAPH DB - NEO4J)")
    print("-" * 50)
    # Ở đây dùng hàm xây dựng đồ thị có chống Rate Limit
    try:
        index = build_knowledge_graph(PROJECT_ROOT)
    except Exception as e:
        print(f"❌ Lỗi khi xây dựng đồ thị: {e}")
        print("Có thể Neo4j chưa được bật hoặc thông tin đăng nhập trong .env chưa đúng.")
        sys.exit(1)
        
    print("\n" + "=" * 50)
    print("🔍 THỬ NGHIỆM TRUY VẤN TỪ NÃO TRÁI (CYPHER QUERY)")
    print("-" * 50)
    
    config = load_config(PROJECT_ROOT)
    graph_store = create_graph_store(
        config.neo4j_uri, config.neo4j_username, config.neo4j_password, config.neo4j_database
    )
    
    # Test cypher trực tiếp (Ví dụ test Aspirin và Warfarin nếu đã được crawl)
    drug_a = "aspirin"
    drug_b = "warfarin"
    print(f"Hỏi: {drug_a} có tương tác với {drug_b} không?")
    
    relations = exact_interaction_query(graph_store, drug_a, drug_b)
    if relations:
        print(f"🤖 ĐÁP ÁN: CÓ! Quan hệ là: {relations}")
    else:
        print(f"🤖 ĐÁP ÁN: KHÔNG TÌM THẤY dữ liệu tương tác trong Đồ thị Não Trái.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
