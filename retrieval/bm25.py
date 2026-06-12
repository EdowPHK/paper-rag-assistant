from typing import List
from collections import Counter
import math
import re

from schemas import PdfChunk, RetrievalResult


_RETRIEVAL_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def _tokenize_for_retrieval(text: str) -> List[str]:
    return [token.lower() for token in _RETRIEVAL_TOKEN_RE.findall(text)]


def bm25_search_chunks(
    query: str,
    chunks: List[PdfChunk],
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not chunks:
        return []

    tokenized_docs = [_tokenize_for_retrieval(chunk["text"]) for chunk in chunks]
    query_terms = _tokenize_for_retrieval(query)
    if not query_terms:
        return []

    doc_count = len(tokenized_docs)
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / doc_count
    document_frequency: Counter[str] = Counter()
    for doc in tokenized_docs:
        document_frequency.update(set(doc))

    scored: List[RetrievalResult] = []
    for chunk, doc_tokens in zip(chunks, tokenized_docs, strict=True):
        if not doc_tokens:
            continue

        term_frequency = Counter(doc_tokens)
        score = 0.0
        doc_len = len(doc_tokens)
        for term in query_terms:
            tf = term_frequency.get(term, 0)
            if not tf:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf * (tf * (k1 + 1)) / denominator

        if score <= 0:
            continue
        scored.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "heading": chunk["heading"],
            "text": chunk["text"],
            "score": score,
            "bm25_score": score,
            "retrieval_source": "bm25",
        })

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]
