import streamlit as st
import os

@st.cache_resource
def get_agent():
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ner_graph.router_agent import get_router_query_engine
    # Project root is one level up from ui/ folder
    return get_router_query_engine(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def render_knowledge_tab():
    st.header("🧠 Knowledge Assistant")
    st.write("GraphRAG + VectorRAG Router Agent")
    
    # Init session state cho chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Nạp Agent (dùng st.cache_resource hoặc lazy load)
    if "agent" not in st.session_state:
        st.session_state.agent = None

    # Nút khởi tạo Agent thủ công để không làm treo UI lúc đầu
    if st.session_state.agent is None:
        if st.button("🚀 Start Router Agent"):
            with st.spinner("Đang kết nối Não Trái (Neo4j) và Não Phải (ChromaDB)..."):
                st.session_state.agent = get_agent()
            st.success("✅ Hệ thống đã sẵn sàng!")
            st.rerun()
    else:
        # Hiển thị lịch sử chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "thought" in msg:
                    with st.expander("⚙️ System Thought Process"):
                        st.write(f"**✅ Não được chọn:** `{msg['tool']}`")
                        st.write(f"**💡 Lý do:** {msg['thought']}")

        # Khung nhập chat
        if prompt := st.chat_input("Hỏi tôi về Aspirin, Warfarin hoặc bất kỳ loại thuốc nào..."):
            # Thêm tin nhắn của user vào lịch sử
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Đang phân tích câu hỏi để chọn Não..."):
                    from llama_index.core import QueryBundle
                    
                    agent = st.session_state.agent
                    bundle = QueryBundle(prompt)
                    
                    # BƯỚC 1: Chủ động gọi Selector để lấy bằng được "Luồng Suy Nghĩ"
                    selector_result = agent._selector.select(agent._metadatas, bundle)
                    
                    selected_tool = "Unknown"
                    reason = "No reason provided"
                    selected_idx = 0
                    
                    if selector_result and hasattr(selector_result, "selections") and len(selector_result.selections) > 0:
                        selection = selector_result.selections[0]
                        selected_idx = selection.index
                        selected_tool = agent._metadatas[selected_idx].name if hasattr(agent._metadatas[selected_idx], "name") else str(selected_idx)
                        
                        # Việt hoá tên
                        if "vector" in selected_tool.lower() or "0" in str(selected_idx):
                            selected_tool = "Não Phải (Lý thuyết / Vector)"
                        else:
                            selected_tool = "Não Trái (Logic / Graph)"
                            
                        reason = selection.reason
                    
                    # BƯỚC 2: Hiển thị WOW Factor ngay lập tức
                    with st.expander("⚙️ System Thought Process (Quá trình suy luận của Agent)", expanded=True):
                        st.write(f"**🧭 Phân tích câu hỏi:** {prompt}")
                        st.write(f"**✅ Não được chọn:** `{selected_tool}`")
                        st.write(f"**💡 Lý do:** {reason}")

                with st.spinner(f"Đang truy xuất dữ liệu từ {selected_tool}..."):
                    # BƯỚC 3: Truy vấn thẳng vào engine đã được chọn
                    chosen_engine = agent._query_engines[selected_idx]
                    response = chosen_engine.query(bundle)
                        
                    # Lưu tin nhắn AI
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": str(response),
                        "tool": selected_tool,
                        "thought": reason
                    })
                        
                    st.success(str(response))
