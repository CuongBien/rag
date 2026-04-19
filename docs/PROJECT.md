# NER — Tóm tắt dự án và công việc đã làm

## Mục đích

Dự án **NER** là một pipeline thử nghiệm **Graph RAG**: đọc tài liệu (PDF), trích xuất đồ thị tri thức bằng LLM, lưu vào **Neo4j**, gộp thực thể trùng nghĩa, rồi **truy vấn** bằng LlamaIndex `PropertyGraphIndex`.

LLM dùng **Groq** qua API tương thích OpenAI (không dùng gói `llama-index-llms-groq` do xung đột phiên bản với `llama-index-llms-openai`).

---

## Công nghệ

| Thành phần | Vai trò |
|------------|---------|
| **uv** | Quản lý môi trường Python và dependencies (`pyproject.toml`, `uv.lock`) |
| **Python ≥ 3.12** | Phiên bản yêu cầu trong `pyproject.toml` |
| **LlamaIndex** | `PropertyGraphIndex`, `SimpleLLMPathExtractor`, `ImplicitPathExtractor` |
| **Neo4j** | Lưu property graph (`Neo4jPropertyGraphStore`) |
| **Groq** | Chat completions qua `https://api.groq.com/openai/v1` |
| **pymupdf4llm** | Chuyển PDF trong thư mục `data/` sang markdown để đưa vào index |
| **sentence-transformers / BGE-M3** | Embedding local (`BAAI/bge-m3`): vector cho **Chunk** và **Entity** trong Neo4j khi `embed_kg_nodes=True` |

---

## Cấu trúc thư mục (phần chính)

```
NER/
├── pyproject.toml          # Dependencies (uv)
├── graph_rag_test.py       # Entry: gọi pipeline
├── data/                   # PDF đầu vào
├── ner_graph/              # Module tách nhỏ
│   ├── __init__.py
│   ├── config.py           # Đọc .env, AppConfig
│   ├── llm_client.py       # GroqOpenAI (metadata + client OpenAI-compatible)
│   ├── graph_store.py      # Neo4j + SafeNeo4jPropertyGraphStore + sanitize name
│   ├── ingest.py           # Load PDF → Document
│   ├── embeddings.py       # HuggingFace BGE-M3 → Settings.embed_model
│   ├── entity_merge.py     # So khớp + LLM xác nhận + merge node (APOC)
│   └── pipeline.py         # Nối toàn bộ bước
├── pdf_gen.py              # Tiện ích PDF (không thuộc pipeline Graph RAG chính)
└── main.py                 # Placeholder hello (không dùng cho Graph RAG)
```

---

## Luồng xử lý (`ner_graph.pipeline.run_pipeline`)

1. **Config** — `load_config`: đọc biến môi trường, đường dẫn `data/` = `{project_root}/data`.
2. **LLM** — `create_llm`: client Groq qua class `GroqOpenAI` (ghi đè `metadata` để không phụ thuộc bảng model OpenAI của LlamaIndex).
3. **Embed (BGE-M3)** — `create_bge_m3_embed_model`: `Settings.embed_model`; dùng cho **chunk** + **entity** khi build index (`embed_kg_nodes=True`). Vector được ghi vào Neo4j (`db.create.setNodeVectorProperty`). Chỉ mục vector mặc định của store là trên `__Entity__`; embedding chunk vẫn lưu trên node Chunk để các bước sau (multi-hop + MENTIONS) dùng.
4. **Neo4j** — `create_graph_store`: kết nối; `refresh_schema=False` để tránh lỗi schema khi dữ liệu cũ có kiểu lạ.
5. **Sanitize** — `sanitize_entity_names`: sửa node `__Entity__` có `name` dạng list (hậu quả merge cũ) → string.
6. **Ingest** — `load_documents_from_data_dir`: mọi `**/*.pdf` trong `data/`.
7. **Index** — `PropertyGraphIndex.from_documents` với `embed_kg_nodes=True` (embedding chunk + entity bằng BGE-M3, lưu Neo4j).
8. **Merge entity** — `merge_similar_entities`: chuẩn hóa tên, blocking, điểm tương đồng, LLM YES/NO, merge bằng `apoc.refactor.mergeNodes` + `SET node.name`.
9. **Query** — `as_query_engine(include_text=True)` và một câu hỏi mẫu (có thể đổi trong `pipeline.py`).

---

## Biến môi trường (`.env`)

| Biến | Ý nghĩa |
|------|---------|
| `GROQ_API_KEY` | Bắt buộc |
| `GROQ_MODEL` | Mặc định gợi ý: `llama-3.3-70b-versatile` |
| `NEO4J_URI` | Ví dụ `bolt://127.0.0.1:7687` |
| `NEO4J_USERNAME` | Thường `neo4j` |
| `NEO4J_PASSWORD` | Bắt buộc |
| `MERGE_SIMILARITY_THRESHOLD` | Ngưỡng fuzzy (mặc định ~0.6) |
| `MERGE_MAX_LLM_CHECKS` | Giới hạn số lần hỏi LLM khi merge |
| `EMBED_MODEL_NAME` | Mặc định `BAAI/bge-m3` |
| `EMBED_BATCH_SIZE` | Batch embedding (mặc định `8`, giảm nếu thiếu RAM/VRAM) |
| `EMBED_DEVICE` | Ví dụ `cuda`, `cpu`; để trống = tự chọn. Nếu ghi `cuda` nhưng PyTorch chỉ bản **CPU**, code sẽ tự **fallback `cpu`** (cần cài PyTorch bản CUDA từ pytorch.org nếu muốn GPU). |
| `EMBED_TRUST_REMOTE_CODE` | `true`/`false` — chỉ bật khi tin tưởng model trên Hub |

---

## Chạy thử

```bash
cd d:\NER
uv sync
uv run python graph_rag_test.py
```

Cần **Neo4j đang chạy** và plugin **APOC** nếu dùng bước merge (`apoc.refactor.mergeNodes`).

---

## Công việc đã thực hiện (theo tiến trình phát triển)

1. **Môi trường uv** — Khởi tạo project, thêm dependencies: LlamaIndex, Neo4j driver, `python-dotenv`, sau đó bổ sung `pymupdf4llm`, v.v.
2. **Groq + LlamaIndex** — Dùng `OpenAI` của LlamaIndex với `api_base` Groq; subclass `GroqOpenAI` để `metadata` không gọi `openai_modelname_to_contextsize` cho model không phải GPT.
3. **Neo4j** — `Neo4jPropertyGraphStore`, xử lý lỗi kết nối (refused), URI `bolt`/`neo4j`.
4. **Rate limit Groq** — Điều chỉnh workers, chunk, giới hạn tài liệu/ký tự (trong các phiên bản trước khi chuyển sang module).
5. **Entity merge** — Logic gộp thực thể tương tự; sửa merge để `name` luôn là **string** (không `properties: combine` gây `StringArray`).
6. **Schema / sanitize** — `SafeNeo4jPropertyGraphStore.get_schema` bắt `CypherTypeError`; sanitize `name` list trong DB không phụ thuộc `apoc.meta.type` khi không có APOC meta.
7. **Refactor** — Tách `graph_rag_test.py` thành package `ner_graph/` (config, LLM, graph store, ingest, entity merge, pipeline).
8. **Chunk + embedding BGE-M3** — `ner_graph/embeddings.py`, `Settings.embed_model`, `embed_kg_nodes=True`: vector chunk + entity trong Neo4j (unified store).

---

## Lưu ý vận hành

- **Groq TPM**: Tier miễn phí có giới hạn token/phút; nếu 429, giảm `max_paths_per_chunk`, số PDF, hoặc chờ.
- **BGE-M3 / PyTorch**: Lần đầu chạy sẽ tải model (nặng); có thể đặt `EMBED_DEVICE=cuda` nếu có GPU. Đổi model embedding sau khi đã tạo vector index trên Neo4j có thể cần xóa index/graph và ingest lại cho đúng chiều vector.
- **Dữ liệu graph cũ lỗi**: Có thể `MATCH (n) DETACH DELETE n` trong Neo4j Browser rồi chạy lại pipeline, hoặc chỉ sửa node `__Entity__` có `name` kiểu list.
- **Câu hỏi mẫu**: Hiện hard-code trong `ner_graph/pipeline.py` — đổi trực tiếp hoặc tách thêm CLI/env nếu cần.

---

## File tài liệu này

- Đường dẫn: `docs/PROJECT.md`
- Mục đích: một chỗ tóm tắt kiến trúc, cách chạy và lịch sử công việc chính của repo.
