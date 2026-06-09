from sentence_transformers import SentenceTransformer       # Embedding model
from typing import List, Tuple, Dict, TypedDict
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams
import hashlib
import json
import pymupdf
import logging
import os
import re

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+|(?<=\n)\n+")
_ENCODER_CACHE: Dict[str, SentenceTransformer] = {}
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_CONFIG_CACHE: Dict[str, object] = {}

class PdfPage(TypedDict):
    source: str
    page_id: int
    text: str | None


class PdfSentence(TypedDict):
    source: str
    page_id: int
    heading: str
    sentence: str


class PdfChunk(TypedDict):
    source: str
    chunk_id: str
    chunk_index: int
    page_start: int
    page_end: int
    heading: str
    text: str


class EmbeddedPdfSentence(TypedDict):
    source: str
    page_id: int
    heading: str
    sentence: str
    embedding: List[float]


class EmbeddedPdfChunk(TypedDict):
    source: str
    chunk_id: str
    chunk_index: int
    page_start: int
    page_end: int
    heading: str
    text: str
    embedding: List[float]


class IndexResult(TypedDict):
    source: str
    pages: int
    chunks: int
    collection_name: str


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
        "chunk_target_tokens": int(raw.get("chunk_target_tokens", 220)),
        "chunk_overlap_tokens": int(raw.get("chunk_overlap_tokens", 40)),
    }
    return _CONFIG_CACHE

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

    encoder = _get_encoder(model)
    vector_size = encoder.get_embedding_dimension()

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=models.Distance.DOT),
    )

def parse_pdf_to_pages(pdf_path: str) -> List[PdfPage]:
    try:
        source_name = os.path.basename(pdf_path)
        with pymupdf.open(pdf_path) as doc:
            pages: List[PdfPage] = []
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text and text.strip():
                    text = text.strip()
                pages.append({
                    "source": source_name,
                    "page_id": index,
                    "text": text,
                })
            return pages
    except (ValueError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception("Failed to parse PDF: %s", pdf_path)
        raise ValueError(f"Cannot parse PDF: {pdf_path}") from exc

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

def split_text_into_headings_and_sentence(file_path: str) -> List[Tuple[str, str]]:
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

def split_pdf_pages_into_sentences(pages: List[PdfPage]) -> List[PdfSentence]:
    items: List[PdfSentence] = []
    current_heading = ""

    for page in pages:
        page_text = page.get("text") or ""
        for line in page_text.splitlines():
            if is_heading_title(line):
                current_heading = line.strip("#").strip()
                continue
            for sentence in split_into_sentences(line):
                items.append(
                    {
                        "source": page["source"],
                        "page_id": page["page_id"],
                        "heading": current_heading,
                        "sentence": sentence,
                    }
                )

    return items

def _make_chunk_id(source: str, page_start: int, page_end: int, heading: str, text: str) -> str:
    raw = f"{source}:{page_start}:{page_end}:{heading}:{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def split_pdf_pages_into_chunks(
    pages: List[PdfPage],
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
    model: str | None = None,
) -> List[PdfChunk]:
    cfg = get_config()
    target_tokens = target_tokens or int(cfg.get("chunk_target_tokens", 220))
    overlap_tokens = overlap_tokens if overlap_tokens is not None else int(cfg.get("chunk_overlap_tokens", 40))
    if target_tokens <= 0:
        raise ValueError("target_tokens must be a positive integer")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be a non-negative integer")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    if not pages:
        return []

    model = str(model or cfg.get("embed_model"))
    tokenizer = _get_encoder(model).tokenizer

    chunks: List[PdfChunk] = []
    current_heading: str = ""

    chunk_token_ids: List[int] = []
    chunk_page_start: int | None = None
    chunk_page_end: int | None = None
    chunk_heading: str = ""

    def decode(token_ids: List[int]) -> str:
        return tokenizer.decode(token_ids).strip()

    def append_chunk(token_ids: List[int], page_start: int, page_end: int, heading: str) -> None:
        text = decode(token_ids)
        if not text:
            return
        source = pages[0]["source"]
        chunks.append({
            "source": source,
            "chunk_id": _make_chunk_id(source, page_start, page_end, heading, text),
            "chunk_index": len(chunks),
            "page_start": page_start,
            "page_end": page_end,
            "heading": heading,
            "text": text,
        })

    for page in pages:
        page_id = page["page_id"]
        text = page.get("text") or ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            if is_heading_title(line):
                current_heading = line.strip("#").strip()
                continue

            tokenized_line = tokenizer.encode(line + "\n", add_special_tokens=False)
            if not tokenized_line:
                continue

            if len(tokenized_line) > target_tokens:
                if chunk_token_ids and chunk_page_start is not None and chunk_page_end is not None:
                    append_chunk(chunk_token_ids, chunk_page_start, chunk_page_end, chunk_heading)
                    chunk_token_ids = []
                    chunk_page_start = None
                    chunk_page_end = None

                start = 0
                while start < len(tokenized_line):
                    end = min(start + target_tokens, len(tokenized_line))
                    append_chunk(tokenized_line[start:end], page_id, page_id, current_heading)
                    if end == len(tokenized_line):
                        break
                    start = max(0, end - overlap_tokens)
                continue

            if chunk_page_start is None:
                chunk_page_start = page_id
                chunk_heading = current_heading

            if chunk_token_ids and len(chunk_token_ids) + len(tokenized_line) > target_tokens:
                previous_page_end = chunk_page_end or page_id
                append_chunk(chunk_token_ids, chunk_page_start, chunk_page_end or page_id, chunk_heading)
                chunk_token_ids = chunk_token_ids[-overlap_tokens:] if overlap_tokens else []
                chunk_page_start = previous_page_end if chunk_token_ids else page_id
                chunk_page_end = page_id
                chunk_heading = current_heading
                if chunk_token_ids and len(chunk_token_ids) + len(tokenized_line) > target_tokens:
                    chunk_token_ids = []
                    chunk_page_start = page_id

            chunk_token_ids.extend(tokenized_line)
            chunk_page_end = page_id

    if chunk_token_ids and chunk_page_start is not None and chunk_page_end is not None:
        append_chunk(chunk_token_ids, chunk_page_start, chunk_page_end, chunk_heading)

    return chunks
    

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
) -> List[Tuple[str, str, List[float]]]:
    """Embed a list of (heading, sentence) pairs and return
    a list of tuples (heading, sentence, embedding).
    """
    model = model or get_config().get("embed_model")
    encoder = _get_encoder(model)

    sentence_items = [(heading, sentence) for heading, sentence in lines if sentence]
    sentences = [s for (_, s) in sentence_items]
    sentence_embeddings = _embed_items(encoder, sentences)

    return [
        (heading, sentence, embedding)
        for (heading, sentence), (_, embedding) in zip(sentence_items, sentence_embeddings, strict=True)
    ]

def embed_pdf_texts(
    lines: List[PdfSentence],
    model: str | None = None,
) -> List[EmbeddedPdfSentence]:
    model = model or get_config().get("embed_model")
    encoder = _get_encoder(model)

    sentence_items = [item for item in lines if item["sentence"]]
    sentences = [item["sentence"] for item in sentence_items]
    embeddings = _embed_items(encoder, sentences)

    return [
        {
            "source": item["source"],
            "page_id": item["page_id"],
            "heading": item["heading"],
            "sentence": sentence,
            "embedding": embedding,
        }
        for item, (sentence, embedding) in zip(sentence_items, embeddings, strict=True)
    ]

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


def embed_pdf_chunks(
    chunks: List[PdfChunk],
    model: str | None = None,
) -> List[EmbeddedPdfChunk]:
    model = str(model or get_config().get("embed_model"))
    encoder = _get_encoder(model)
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

def upsert_pdf_sentences(
    items: List[EmbeddedPdfSentence],
    collection_name: str | None = None,
) -> None:
    if not items:
        return
    
    cfg = get_config()
    collection_name = str(collection_name or cfg.get("collection_name"))
    _create_collection(collection_name, str(cfg.get("embed_model")))
    client = _get_qdrant_client(cfg)

    points = [
        models.PointStruct(
            id=f'{item["source"]}:{item["page_id"]}:{index}',
            vector=item["embedding"],
            payload={
                "source": item["source"],
                "page_id": item["page_id"],
                "heading": item["heading"],
                "sentence": item["sentence"],
            },
        )
        for index, item in enumerate(items)
    ]

    client.upsert(collection_name=collection_name, points=points)


def index_pdf(pdf_path: str, collection_name: str | None = None, model: str | None = None) -> IndexResult:
    cfg = get_config()
    collection_name = str(collection_name or cfg.get("collection_name"))
    model = str(model or cfg.get("embed_model"))
    pages = parse_pdf_to_pages(pdf_path)
    chunks = split_pdf_pages_into_chunks(pages, model=model)
    embedded_items = embed_pdf_chunks(chunks, model=model)
    upsert_pdf_chunks(embedded_items, collection_name=collection_name, model=model)
    return {
        "source": os.path.basename(pdf_path),
        "pages": len(pages),
        "chunks": len(embedded_items),
        "collection_name": collection_name,
    }
