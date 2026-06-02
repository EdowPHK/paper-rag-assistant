from sentence_transformers import SentenceTransformer       # Embedding model
from typing import List, Tuple, Dict
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams
import json
import requests
import logging
import os
import re

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+|(?<=\n)\n+")
_ENCODER_CACHE: Dict[str, SentenceTransformer] = {}
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_CONFIG_CACHE: Dict[str, object] = {}

def _load_config(path: str = _CONFIG_PATH) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Missing config file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {path}") from exc

def _get_qdrant_client(config: Dict[str, str]) -> QdrantClient:
    url = config.get("qdrant_url", "").strip()
    api_key = config.get("qdrant_api_key", "").strip()
    if not url or not api_key:
        raise ValueError("qdrant_url and qdrant_api_key must be set in config.json")
    return QdrantClient(url=url, api_key=api_key)

def _get_encoder(model_name: str) -> SentenceTransformer:
    encoder = _ENCODER_CACHE.get(model_name)
    if encoder is None:
        try:
            encoder = SentenceTransformer(model_name)
        except Exception as exc:
            raise ValueError(f"Failed to load SentenceTransformer model: {model_name}") from exc
        _ENCODER_CACHE[model_name] = encoder
    return encoder

def get_config() -> Dict[str, object]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE:
        return _CONFIG_CACHE
    try:
        raw = _load_config()
    except ValueError:
        raw = {}

    _CONFIG_CACHE = {
        "qdrant_url": raw.get("qdrant_url", ""),
        "qdrant_api_key": raw.get("qdrant_api_key", ""),
        "collection_name": raw.get("collection_name", "knowledge_base"),
        "embed_model": raw.get("embed_model", "all-MiniLM-L6-v2"),
        "embed_text_batch_size": int(raw.get("embed_text_batch_size", 32)),
    }
    return _CONFIG_CACHE

def split_into_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]

def is_heading_title(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return False
    if s.startswith("#") and len(s) > 1:
        return True
    if len(s) <= 50 and s.upper() == s and re.search(r"[A-Z]", s):
        return True
    if re.match(r"^\d+(\.\d+)*\s+\S+", s):
        return True
    return False

def split_file_into_headings_and_sentence(file_path: str) -> List[Tuple[str, str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError as exc:
        logging.getLogger(__name__).exception("Failed to read file: %s", file_path)
        raise ValueError(f"Cannot read file: {file_path}") from exc

    pairs: List[Tuple[str, str]] = []
    current_heading = ""

    for line in text.splitlines():
        if is_heading_title(line):
            current_heading = line.strip("#").strip()
            continue
        for sentence in split_into_sentences(line):
            pairs.append((current_heading, sentence))

    return pairs

def _embed_items(
    encoder: SentenceTransformer,
    items: List[str],
) -> List[Tuple[str, List[float]]]:
    if not items:
        return []
    batch = get_config().get("embed_text_batch_size", 32)
    vectors = encoder.encode(items, batch_size=batch, normalize_embeddings=True)
    return [(item, vector.tolist()) for item, vector in zip(items, vectors, strict=True)]

def embed_texts(
    lines: List[Tuple[str, str]],
    model: str | None = None,
) -> Tuple[List[Tuple[str, List[float]]], List[Tuple[str, List[float]]]]:
    model = model or get_config().get("embed_model")
    encoder = _get_encoder(model)

    headings = [heading for heading, _ in lines if heading]
    sentences = [sentence for _, sentence in lines if sentence]

    heading_embeddings = _embed_items(encoder, headings)
    sentence_embeddings = _embed_items(encoder, sentences)

    return heading_embeddings, sentence_embeddings

def fixed_size_chunks(file_path: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as exc:
        logging.getLogger(__name__).exception("Failed to read file: %s", file_path)
        raise ValueError(f"Cannot read file: {file_path}") from exc
    
    return [text[i:i+chunk_size] for i in range(0, len(text), step)]

def embed_chunks(
        chunks: List[str], 
        model: str | None = None,
        ) -> List[Tuple[str, List[float]]]:
    model = model or get_config().get("embed_model")
    encoder = _get_encoder(model)
    return _embed_items(encoder, chunks)

def _create_collection(collection_name: str | None = None, model: str | None = None) -> None:
    cfg = get_config()
    collection_name = collection_name or cfg.get("collection_name")
    model = model or cfg.get("embed_model")
    client = _get_qdrant_client(cfg)

    try:
        client.get_collection(collection_name=collection_name)
        return
    except Exception:
        pass

    encoder = _get_encoder(model)
    vector_size = encoder.get_embedding_dimension()

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )