"""
ingest.py — one-time script to load the knowledge base into ChromaDB.

Run once (or whenever you add new documents):
    python ingest.py

Documents are loaded from:  knowledge_base/**/*.pdf
ChromaDB is written to:     chroma_db/
"""

import os
from pathlib import Path

import pypdf
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CHROMA_PATH        = str(BASE_DIR / "chroma_db")
COLLECTION_NAME    = "finbot_knowledge"

# ── Chunking config ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


def load_documents():
    """Walk knowledge_base/ and load every PDF, tagging source metadata."""
    all_docs = []
    pdf_files = sorted(KNOWLEDGE_BASE_DIR.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under {KNOWLEDGE_BASE_DIR}")

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.relative_to(BASE_DIR)}")
        reader = pypdf.PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                all_docs.append(Document(
                    page_content=text,
                    metadata={
                        "source_file":   pdf_path.name,
                        "source_folder": pdf_path.parent.name,
                        "page":          page_num,
                    },
                ))

    return all_docs


def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY is not set in your .env file.")

    # 1 — Load PDFs
    print(f"\n[1/3] Loading PDFs from {KNOWLEDGE_BASE_DIR} ...")
    docs = load_documents()
    print(f"      Loaded {len(docs)} pages total.\n")

    # 2 — Split into chunks
    print("[2/3] Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"      Created {len(chunks)} chunks.\n")

    # 3 — Embed and store
    print("[3/3] Embedding and storing in ChromaDB ...")
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-1b-v2",
        api_key=api_key,
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},  # scores in [0,1] for threshold filter
    )

    print(f"      Done! {db._collection.count()} chunks stored in {CHROMA_PATH}\n")
    print("Knowledge base is ready. You can now run: python app.py")


if __name__ == "__main__":
    main()
