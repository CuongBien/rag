import pathlib

import pymupdf4llm
from llama_index.core import Document


def load_documents_from_data_dir(data_dir: str) -> list[Document]:
    # data_dir: folder recursively scanned for *.pdf; each file becomes one Document.
    print(f"[ingest] Reading documents from: {data_dir}")
    documents: list[Document] = []
    for pdf_path in pathlib.Path(data_dir).glob("**/*.pdf"):
        markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
        if markdown_text.strip() == "":
            print(f"[ingest] Skip empty text file: {pdf_path.name}")
            continue
        documents.append(
            Document(
                text=markdown_text,
                metadata={"file_name": pdf_path.name, "file_path": str(pdf_path)},
            )
        )
        print(f"[ingest] Loaded {pdf_path.name}: {len(markdown_text)} chars")
    if len(documents) == 0:
        raise RuntimeError(f"No readable documents found in: {data_dir}")
    print(f"[ingest] Total loaded documents: {len(documents)}")
    return documents
