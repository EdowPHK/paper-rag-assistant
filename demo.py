from sentence_transformers import SentenceTransformer       # Embedding model
from typing import List, Tuple, Dict
import logging
import re

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+|(?<=\n)\n+")
_ENCODER_CACHE: Dict[str, SentenceTransformer] = {}
EMBED_TEXT_BATCH_SIZE = 32

def _get_encoder(model_name: str) -> SentenceTransformer:
    encoder = _ENCODER_CACHE.get(model_name)
    if encoder is None:
        try:
            encoder = SentenceTransformer(model_name)
        except Exception as exc:
            raise ValueError(f"Failed to load SentenceTransformer model: {model_name}") from exc
        _ENCODER_CACHE[model_name] = encoder
    return encoder

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


def embed_texts(
    lines: List[Tuple[str, str]],
    model: str = "all-MiniLM-L6-v2",
) -> Tuple[List[Tuple[str, List[float]]], List[Tuple[str, List[float]]]]:
    encoder = _get_encoder(model)
    heading_embeddings: List[Tuple[str, List[float]]] = []
    sentence_embeddings: List[Tuple[str, List[float]]] = []

    headings = [heading for heading, _ in lines if heading]
    sentences = [sentence for _, sentence in lines if sentence]

    if headings:
        heading_vectors = encoder.encode(
            headings,
            batch_size=EMBED_TEXT_BATCH_SIZE,
            normalize_embeddings=True,
        )
        for heading, vector in zip(headings, heading_vectors, strict=True):
            heading_embeddings.append((heading, vector.tolist()))

    if sentences:
        sentence_vectors = encoder.encode(
            sentences,
            batch_size=EMBED_TEXT_BATCH_SIZE,
            normalize_embeddings=True,
        )
        for sentence, vector in zip(sentences, sentence_vectors, strict=True):
            sentence_embeddings.append((sentence, vector.tolist()))

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

def embed_chunks(chunklist: List[str], )