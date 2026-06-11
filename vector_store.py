from typing import List, Dict
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams

from config import get_config
from embeddings import get_encoder
from schemas import EmbeddedPdfChunk

def _get_qdrant_client(config: Dict[str, str]) -> QdrantClient:
    url = config.get("qdrant_url", "").strip()
    api_key = config.get("qdrant_api_key", "").strip()
    if not url or not api_key:
        raise ValueError("qdrant_url and qdrant_api_key must be set in config.json")
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
