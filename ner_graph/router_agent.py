import os
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

from ner_graph.config import load_config
from ner_graph.llm_client import create_llm
from ner_graph.vector_store import get_vector_query_engine
from ner_graph.pipeline import build_query_engine as get_graph_query_engine

def get_router_query_engine(project_root: str):
    """
    Khởi tạo và trả về RouterQueryEngine.
    """
    # Xoá tạm biến môi trường Gemini để hàm create_llm buộc phải chọn Groq
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
        
    config = load_config(project_root)
    
    from llama_index.core import Settings
    llm = create_llm(config)
    Settings.llm = llm
    
    # Khởi tạo Vector Engine (Não Phải)
    vector_query_engine = get_vector_query_engine(project_root, llm=llm)
    
    # Khởi tạo Graph Engine (Não Trái)
    graph_query_engine = get_graph_query_engine(project_root)

    print("[router_agent] Defining tools...")
    # BƯỚC 1: ĐỊNH NGHĨA CÁC BÁN CẦU NÃO THÀNH CÁC "CÔNG CỤ" (TOOLS)
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Chỉ sử dụng công cụ này khi người dùng hỏi về: thông tin chung, "
            "lịch sử ra đời, định nghĩa, tính chất vật lý, cơ chế hoạt động độc lập, "
            "hoặc các thông tin bách khoa về một loại thuốc/hóa chất."
        ),
    )

    graph_tool = QueryEngineTool.from_defaults(
        query_engine=graph_query_engine,
        description=(
            "SỬ DỤNG CÔNG CỤ NÀY ĐẦU TIÊN VÀ BẮT BUỘC khi người dùng hỏi về: "
            "SỰ TƯƠNG TÁC giữa 2 hay nhiều loại thuốc, việc 'uống chung', "
            "'dùng kết hợp', hoặc hệ quả khi pha trộn các hóa chất với nhau."
        ),
    )

    print("[router_agent] Assembling RouterQueryEngine...")
    # BƯỚC 2: KHỞI TẠO ROUTER AGENT (NHẠC TRƯỞNG)
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[vector_tool, graph_tool],
        verbose=False
    )
    
    return router_query_engine
