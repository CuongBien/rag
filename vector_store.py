# Tạo file test_vector.py trong d:\NER\
from ner_graph.vector_store import build_vector_index

# Quá trình này sẽ đọc 54 file txt, nhúng qua model BGE-M3 và lưu vào ChromaDB
index = build_vector_index(".")

# Sau đó mình thử truy vấn Não phải
query_engine = index.as_query_engine()
response = query_engine.query("What is the mechanism of action of Omeprazole?")
print("Trí nhớ Não phải trả lời:\n", response)