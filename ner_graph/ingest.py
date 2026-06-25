import pathlib

import pymupdf4llm
from llama_index.core import Document


def load_documents_from_data_dir(data_dir: str) -> list[Document]:
    print(f"[ingest] Reading documents from: {data_dir}")
    documents: list[Document] = []
    
    # Đọc file PDF (Code cũ)
    for pdf_path in pathlib.Path(data_dir).glob("**/*.pdf"):
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
    for txt_path in pathlib.Path(data_dir).glob("**/*.txt"):
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
        raise RuntimeError(f"No readable documents found in: {data_dir}")
    print(f"[ingest] Total loaded documents: {len(documents)}")
    return documents
