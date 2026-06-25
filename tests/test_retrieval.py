import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from ner_graph.vector_store import build_vector_index
from llama_index.core import Settings
from ner_graph.embeddings import create_bge_m3_embed_model

def main():
    embed_model = create_bge_m3_embed_model(
        "BAAI/bge-m3", 8, "cuda", True, True, False
    )
    Settings.embed_model = embed_model
    index = build_vector_index(PROJECT_ROOT)
    retriever = index.as_retriever(similarity_top_k=15)
    
    nodes = retriever.retrieve("In what year was aspirin discovered?")
    for i, node in enumerate(nodes):
        print(f"--- Node {i+1} Score: {node.score} ---")
        print(node.text[:200].replace("\n", " "))
        if "1853" in node.text or "1897" in node.text:
            print(">>> HISTORICAL DISCOVERY FOUND HERE <<<")

if __name__ == "__main__":
    main()
