from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams

from config import get_config
from embeddings import get_encoder
from schemas import EmbeddedPdfChunk, PdfChunk, RetrievalResult

def _get_qdrant_client(config: Dict[str, str]) -> QdrantClient:
    url = config.get("qdrant_url", "").strip()
    api_key = config.get("qdrant_api_key", "").strip()
    if not url or not api_key:
        raise ValueError("qdrant_url and qdrant_api_key must be set in config.yaml")
    return QdrantClient(url=url, api_key=api_key)

def _create_collection(collection_name: str | None = None, model: str | None = None) -> None:
    cfg = get_config()
    collection_name = str(collection_name or cfg.get("collection_name"))
    model = str(model or cfg.get("embed_model"))
    client = _get_qdrant_client(cfg)

    try:
        client.get_collection(collection_name=collection_name)
        return
    except Exception:
        pass

    encoder = get_encoder(model)
    vector_size = encoder.get_embedding_dimension()

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=models.Distance.DOT),
    )

def upsert_pdf_chunks(
    items: List[EmbeddedPdfChunk],
    collection_name: str | None = None,
    model: str | None = None,
) -> None:
    if not items:
        return

    cfg = get_config()
    collection_name = str(collection_name or cfg.get("collection_name"))
    model = str(model or cfg.get("embed_model"))
    _create_collection(collection_name, model)
    client = _get_qdrant_client(cfg)

    points = [
        models.PointStruct(
            id=item["chunk_id"],
            vector=item["embedding"],
            payload={
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "chunk_index": item["chunk_index"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "heading": item["heading"],
                "text": item["text"],
            },
        )
        for item in items
    ]

    client.upsert(collection_name=collection_name, points=points)

    
def _payload_to_pdf_chunk(payload: Dict[str, Any]) -> PdfChunk:
    return {
        "source": str(payload.get("source", "")),
        "chunk_id": str(payload.get("chunk_id", "")),
        "chunk_index": _payload_int(payload, "chunk_index"),
        "page_start": _payload_int(payload, "page_start"),
        "page_end": _payload_int(payload, "page_end"),
        "heading": str(payload.get("heading", "")),
        "text": str(payload.get("text", "")),
    }


def list_indexed_pdf_chunks(
    collection_name: str | None = None,
    limit: int = 1000,
) -> List[PdfChunk]:
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    cfg = get_config()
    collection_name = str(collection_name or cfg.get("collection_name"))
    client = _get_qdrant_client(cfg)
    if not hasattr(client, "scroll"):
        raise ValueError("Qdrant client does not support scroll")

    chunks: List[PdfChunk] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = getattr(record, "payload", None) or {}
            chunk = _payload_to_pdf_chunk(payload)
            if chunk["chunk_id"] and chunk["text"]:
                chunks.append(chunk)
        if offset is None:
            break

    return chunks

def _payload_int(payload: Dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    if value is None:
        return default
    return int(value)

def _query_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    top_k: int,
) -> List[object]:
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return list(getattr(response, "points", response))

def _payload_to_retrieval_result(
    payload: Dict[str, Any],
    score: float,
    retrieval_source: str,
) -> RetrievalResult:
    return {
        "chunk_id": str(payload.get("chunk_id", "")),
        "source": str(payload.get("source", "")),
        "chunk_index": _payload_int(payload, "chunk_index"),
        "page_start": _payload_int(payload, "page_start"),
        "page_end": _payload_int(payload, "page_end"),
        "heading": str(payload.get("heading", "")),
        "text": str(payload.get("text", "")),
        "score": float(score),
        "retrieval_source": retrieval_source,
    }

def _point_to_retrieval_result(point: object, retrieval_source: str) -> RetrievalResult:
    payload = getattr(point, "payload", None) or {}
    score = float(getattr(point, "score", 0.0) or 0.0)
    result = _payload_to_retrieval_result(payload, score, retrieval_source)
    if not result["chunk_id"]:
        result["chunk_id"] = str(getattr(point, "id", ""))
    return result

def search_vectors(
    query_vector: List[float],
    top_k: int = 5,
    collection_name: str | None = None,
) -> List[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    cfg = get_config()
    collection_name = str(collection_name or cfg.get("collection_name"))

    client = _get_qdrant_client()
    points = _query_qdrant(client, collection_name, query_vector, top_k)

    return [
        _point_to_retrieval_result(point, "vector")
        for point in points
    ]
