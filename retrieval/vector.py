from typing import List

from config import get_config
from embeddings import get_encoder
from schemas import RetrievalResult
from vector_store import search_vectors

def search_pdf_chunks(
    query: str,
    top_k: int = 5,
    collection_name: str | None = None,
    model: str | None = None,
) -> List[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    cfg = get_config()
    model = str(model or cfg.get("embed_model"))

    encoder = get_encoder(model)
    query_vector = encoder.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    return search_vectors(
        query_vector=query_vector,
        top_k=top_k,
        collection_name=collection_name,
    )