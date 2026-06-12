from typing import List

from sentence_transformers import CrossEncoder

from config import get_config
from schemas import PdfChunk, RetrievalResult, RerankedResult
from retrieval.hybrid import hybrid_search_pdf_chunks

_RERANKER_CACHE: dict[str, CrossEncoder] = {}


def get_reranker(model_name: str) -> CrossEncoder:
    reranker = _RERANKER_CACHE.get(model_name)
    if reranker is None:
        try:
            reranker = CrossEncoder(model_name)
        except Exception as exc:
            raise ValueError(f"Failed to load CrossEncoder model: {model_name}") from exc
        _RERANKER_CACHE[model_name] = reranker
    return reranker


def rerank(
    query: str,
    candidates: List[RetrievalResult],
    top_k: int | None = None,
    rerank_model: str | None = None,
) -> List[RerankedResult]:
    cfg = get_config()
    top_k = top_k if top_k is not None else int(cfg.get("rerank_top_k", 5))
    rerank_model = str(rerank_model or cfg.get("rerank_model"))

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not candidates:
        return []

    valid_candidates = [item for item in candidates if item.get("text")]
    if not valid_candidates:
        return []

    reranker = get_reranker(rerank_model)
    pairs = [(query, item["text"]) for item in valid_candidates]
    scores = reranker.predict(pairs)

    reranked: List[RerankedResult] = []
    for item, score in zip(valid_candidates, scores, strict=True):
        result: RerankedResult = {
            "chunk_id": item.get("chunk_id", ""),
            "source": item.get("source", ""),
            "chunk_index": item.get("chunk_index", 0),
            "page_start": item.get("page_start", 0),
            "page_end": item.get("page_end", 0),
            "heading": item.get("heading", ""),
            "text": item.get("text", ""),
            "score": float(score),
            "rerank_score": float(score),
            "retrieval_source": "rerank",
        }
        reranked.append(result)

    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]


def retrieve_with_rerank(
    query: str,
    chunks: List[PdfChunk] | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
    collection_name: str | None = None,
    model: str | None = None,
    rerank_model: str | None = None,
) -> List[RerankedResult]:
    cfg = get_config()
    top_k = top_k if top_k is not None else int(cfg.get("rerank_top_k", 5))
    candidate_k = candidate_k if candidate_k is not None else int(cfg.get("rerank_candidate_k", 20))

    if candidate_k < top_k:
        candidate_k = top_k

    candidates = hybrid_search_pdf_chunks(
        query=query,
        chunks=chunks,
        top_k=candidate_k,
        collection_name=collection_name,
        model=model,
    )

    return rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
        rerank_model=rerank_model,
    )
