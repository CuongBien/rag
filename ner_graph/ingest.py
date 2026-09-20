import pathlib

import pymupdf4llm
from llama_index.core import Document


def load_documents_from_data_dir(data_dir: str) -> list[Document]:
    target_path = pathlib.Path(data_dir)
    if not target_path.exists() or not (any(target_path.glob("**/*.pdf")) or any(target_path.glob("**/*.txt"))):
        fallback = target_path.parent / "data" if target_path.name != "data" else target_path.parent / "data_vector"
        if fallback.exists() and (any(fallback.glob("**/*.pdf")) or any(fallback.glob("**/*.txt"))):
            print(f"[ingest] No documents in '{data_dir}'; automatically switching to '{fallback}'")
            target_path = fallback

    print(f"[ingest] Reading documents from: {target_path}")
    documents: list[Document] = []
    
    # Đọc file PDF (Code cũ)
    for pdf_path in target_path.glob("**/*.pdf"):
        markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
        if markdown_text.strip() == "":
            print(f"[ingest] Skip empty pdf file: {pdf_path.name}")
            continue
        documents.append(
            Document(
                text=markdown_text,
                metadata={"file_name": pdf_path.name, "file_path": str(pdf_path)},
            )
        )
        print(f"[ingest] Loaded {pdf_path.name}: {len(markdown_text)} chars")
        
    # Đọc file TXT (Wikipedia DDI data)
    for txt_path in target_path.glob("**/*.txt"):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip() == "":
                print(f"[ingest] Skip empty txt file: {txt_path.name}")
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={"file_name": txt_path.name, "file_path": str(txt_path)},
                )
            )
            print(f"[ingest] Loaded {txt_path.name}: {len(text)} chars")
        except Exception as e:
            print(f"[ingest] Error reading {txt_path.name}: {e}")

    if len(documents) == 0:
        raise RuntimeError(
            f"No readable documents (.pdf or .txt) found in: {target_path}. "
            "Please place your PDF in 'data/' or run 'uv run python scripts/download_wiki.py' to download drug texts into 'data_vector/'."
        )
    print(f"[ingest] Total loaded documents: {len(documents)}")
    return documents
