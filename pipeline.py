import os

from config import get_config
from schemas import AnswerResult, IndexResult
from parsing import parse_pdf_to_pages
from chunking import split_pdf_pages_into_chunks
from embeddings import embed_pdf_chunks
from vector_store import upsert_pdf_chunks
from generation.answer import answering


def index_pdf(
    pdf_path: str,
    collection_name: str | None = None,
    model: str | None = None,
) -> IndexResult:
    cfg = get_config()
    collection_name = str(collection_name or cfg.get("collection_name"))
    model = str(model or cfg.get("embed_model"))

    pages = parse_pdf_to_pages(pdf_path=pdf_path)
    chunks = split_pdf_pages_into_chunks(
        pages=pages,
        model=model,
    )
    embedded_chunks = embed_pdf_chunks(
        chunks=chunks,
        model=model,
    )
    upsert_pdf_chunks(
        items=embedded_chunks,
        collection_name=collection_name,
        model=model,
    )

    return {
        "source": os.path.basename(pdf_path),
        "pages": len(pages),
        "chunks": len(embedded_chunks),
        "collection_name": collection_name,
    }


def answer_query(
    query: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    collection_name: str | None = None,
    model: str | None = None,
    rerank_model: str | None = None,
    max_context_chars: int | None = None,
) -> AnswerResult:
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    cfg = get_config()
    candidate_k = candidate_k if candidate_k is not None else int(cfg.get("rerank_candidate_k", 20))
    top_k = top_k if top_k is not None else int(cfg.get("rerank_top_k", 5))
    collection_name = str(collection_name or cfg.get("collection_name"))
    model = str(model or cfg.get("embed_model"))
    rerank_model = str(rerank_model or cfg.get("rerank_model"))
    max_context_chars = max_context_chars if max_context_chars is not None else int(cfg.get("max_context_chars", 6000))

    return answering(
        query=query,
        candidate_k=candidate_k,
        embed_model=model,
        rerank_model=rerank_model,
        top_k=top_k,
        collection_name=collection_name,
        max_context_chars=max_context_chars,
    )
