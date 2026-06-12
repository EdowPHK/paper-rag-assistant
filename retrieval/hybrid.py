from typing import List
from schemas import RetrievalResult, PdfChunk

from vector_store import list_indexed_pdf_chunks
from retrieval.bm25 import bm25_search_chunks
from retrieval.rrf import rrf_fuse_results
from retrieval.vector import search_pdf_chunks

def hybrid_search_pdf_chunks(
    query: str,
    chunks: List[PdfChunk] | None = None,
    top_k: int = 5,
    collection_name: str | None = None,
    model: str | None = None,
) -> List[RetrievalResult]:
    if chunks is None:
        chunks = list_indexed_pdf_chunks(collection_name=collection_name)
    vector_results = search_pdf_chunks(
        query=query,
        top_k=top_k,
        collection_name=collection_name,
    )
    bm25_results = bm25_search_chunks(query=query, chunks=chunks, top_k=top_k)
    return rrf_fuse_results([vector_results, bm25_results], top_k=top_k)
