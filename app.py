import streamlit as st
import os

# Ép tắt Telemetry Phoenix để khỏi bị văng timeout 4317
os.environ["PHOENIX_ENABLED"] = "0"

from ui.tab_translation import render_translation_tab
from ui.tab_knowledge import render_knowledge_tab

def main():
    st.set_page_config(page_title="Cheminformatics Pipeline", layout="wide")
    st.title("🧪 Cheminformatics Hybrid Translation Pipeline")
    
    with st.sidebar:
        st.header("⚙️ API Configuration")
        st.markdown("Temporarily test API keys here.")
        gemini_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
        groq_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

    tab1, tab2 = st.tabs(["Chemical Translation", "Knowledge Assistant"])
    
    with tab1:
        render_translation_tab()
    
    with tab2:
        render_knowledge_tab()

if __name__ == "__main__":
    main()
