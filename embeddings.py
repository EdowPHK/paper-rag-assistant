from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple

from config import get_config
from schemas import PdfChunk, EmbeddedPdfChunk

_ENCODER_CACHE: Dict[str, SentenceTransformer] = {}

def get_encoder(model_name: str) -> SentenceTransformer:
    encoder = _ENCODER_CACHE.get(model_name)
    if encoder is None:
        try:
            encoder = SentenceTransformer(model_name)
        except Exception as exc:
            raise ValueError(f"Failed to load SentenceTransformer model: {model_name}") from exc
        _ENCODER_CACHE[model_name] = encoder
    return encoder

def _embed_items(
    encoder: SentenceTransformer,
    items: List[str],
) -> List[Tuple[str, List[float]]]:
    if not items:
        return []
    batch = get_config().get("embed_text_batch_size", 32)
    vectors = encoder.encode(items, batch_size=batch, normalize_embeddings=True)
    return [(item, vector.tolist()) for item, vector in zip(items, vectors, strict=True)]

def embed_chunks(
        chunks: List[str], 
        model: str | None = None,
        ) -> List[Tuple[str, List[float]]]:
    model = str(model or get_config().get("embed_model"))
    encoder = get_encoder(model)
    return _embed_items(encoder, chunks)

def embed_pdf_chunks(
    chunks: List[PdfChunk],
    model: str | None = None,
) -> List[EmbeddedPdfChunk]:
    model = str(model or get_config().get("embed_model"))
    encoder = get_encoder(model)
    chunk_items = [chunk for chunk in chunks if chunk["text"]]
    embeddings = _embed_items(encoder, [chunk["text"] for chunk in chunk_items])
    return [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "heading": chunk["heading"],
            "text": text,
            "embedding": embedding,
        }
        for chunk, (text, embedding) in zip(chunk_items, embeddings, strict=True)
    ]