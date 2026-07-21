"""
chatbot/rag.py — ChromaDB retriever module.

Singleton pattern: DB + embeddings are created once per process.
"""

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

CHROMA_PATH     = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "finbot_knowledge"
SCORE_THRESHOLD = 0.3

_db = None   # singleton — shared across all calls


def _get_db() -> Chroma:
    global _db
    if _db is not None:
        return _db
    if not os.path.isdir(CHROMA_PATH):
        return None
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return None
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-1b-v2",
        api_key=api_key,
    )
    _db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return _db


def retrieve(query: str, k: int = 4) -> tuple:
    """
    Retrieve top-k relevant chunks for a query.
    Returns (context_str, sources_list).
    Uses a singleton DB so embeddings are only initialized once per process.
    """
    db = _get_db()
    if db is None:
        return "", []

    scored_docs = db.similarity_search_with_relevance_scores(query, k=k)
    print(f"[RAG] query={query!r}")
    for doc, score in scored_docs:
        print(f"  score={score:.4f}  file={doc.metadata.get('source_file','?')}  content={doc.page_content[:60]!r}")

    docs = [doc for doc, score in scored_docs if score >= SCORE_THRESHOLD]
    if not docs:
        print(f"  → no docs above threshold {SCORE_THRESHOLD}")
        return "", []

    context_parts, sources, seen = [], [], set()
    for i, doc in enumerate(docs, 1):
        folder = doc.metadata.get("source_folder", "")
        fname  = doc.metadata.get("source_file", "")
        page   = doc.metadata.get("page", 0)
        key    = f"{folder}/{fname}"
        context_parts.append(f"[Source {i}: {folder}/{fname}]\n{doc.page_content}")
        if key not in seen:
            seen.add(key)
            sources.append({"folder": folder, "file": fname, "page": int(page) + 1})

    print(f"  → {len(sources)} unique source(s) returned")
    return "\n\n".join(context_parts), sources


def retrieve_context(query: str, k: int = 4) -> str:
    """Convenience wrapper — returns only the context string."""
    context, _ = retrieve(query, k=k)
    return context
