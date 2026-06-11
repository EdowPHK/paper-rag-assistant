from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import Counter
from typing import Any, List, Tuple, Dict, TypedDict
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams
from openai import OpenAI
import hashlib
import json
import math
import pymupdf
import logging
import os
import re

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+|(?<=\n)\n+")
_RETRIEVAL_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_ENCODER_CACHE: Dict[str, SentenceTransformer] = {}
_RERANKER_CACHE: Dict[str, CrossEncoder] = {}
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


class RetrievalResult(TypedDict, total=False):
    chunk_id: str
    source: str
    chunk_index: int
    page_start: int
    page_end: int
    heading: str
    text: str
    score: float
    vector_score: float
    bm25_score: float
    rrf_score: float
    retrieval_source: str


class RerankedResult(TypedDict):
    chunk_id: str
    source: str
    chunk_index: int
    page_start: int
    page_end: int
    heading: str
    text: str
    score: float
    rerank_score: float
    retrieval_source: str


class AnswerSource(TypedDict, total=False):
    source: str
    chunk_index: int
    page_start: int
    page_end: int
    heading: str
    score: float
    retrieval_source: str


class AnswerResult(TypedDict):
    query: str
    answer: str
    sources: List[AnswerSource]

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

def get_qdrant_apikey() -> str:
    return os.getenv("QDRANT_API_KEY", "").strip()

def _get_encoder(model_name: str) -> SentenceTransformer:
    encoder = _ENCODER_CACHE.get(model_name)
    if encoder is None:
        try:
            encoder = SentenceTransformer(model_name)
        except Exception as exc:
            raise ValueError(f"Failed to load SentenceTransformer model: {model_name}") from exc
        _ENCODER_CACHE[model_name] = encoder
    return encoder

def _get_reranker(model_name: str) -> CrossEncoder:
    reranker = _RERANKER_CACHE.get(model_name)
    if reranker is None:
        try:
            reranker = CrossEncoder(model_name)
        except Exception as exc:
            raise ValueError(f"Failed to load CrossEncoder model:{model_name}") from exc
        _RERANKER_CACHE[model_name] = reranker
    return reranker

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
        "qdrant_api_key": get_qdrant_apikey(),
        "collection_name": raw.get("collection_name", "knowledge_base"),
        "embed_model": raw.get("embed_model", "all-MiniLM-L6-v2"),
        "embed_text_batch_size": int(raw.get("embed_text_batch_size", 32)),
        "chunk_target_tokens": int(raw.get("chunk_target_tokens", 220)),
        "chunk_overlap_tokens": int(raw.get("chunk_overlap_tokens", 40)),
        "rerank_model": str(raw.get("rerank_model", _DEFAULT_RERANK_MODEL)),
        "rerank_candidate_k": int(raw.get("rerank_candidate_k", 20)),
        "rerank_top_k": int(raw.get("rerank_top_k", 5)),
        "max_context_chars": int(raw.get("max_context_chars", 6000)),
        "llm_model": str(raw.get("llm_model", "deepseek-chat")),
        "llm_api_key_env": str(raw.get("llm_api_key_env", "DEEPSEEK_API_KEY")),
        "llm_url": str(raw.get("llm_url", raw.get("llm_base_url", "https://api.deepseek.com"))),
    }
    return _CONFIG_CACHE

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
    model = str(model or get_config().get("embed_model"))
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
    model = str(model or get_config().get("embed_model"))
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
    model = str(model or get_config().get("embed_model"))
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


def _payload_int(payload: Dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    if value is None:
        return default
    return int(value)


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


def search_pdf_chunks(
    query: str,
    top_k: int = 5,
    collection_name: str | None = None,
    model: str | None = None,
) -> List[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    cfg = get_config()
    collection_name = str(collection_name or cfg.get("collection_name"))
    model = str(model or cfg.get("embed_model"))
    encoder = _get_encoder(model)
    query_vector = encoder.encode([query], normalize_embeddings=True)[0].tolist()
    client = _get_qdrant_client(cfg)
    points = _query_qdrant(client, collection_name, query_vector, top_k)
    return [_point_to_retrieval_result(point, "vector") for point in points]


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


def rrf_fuse_results(
    ranked_lists: List[List[RetrievalResult]],
    top_k: int = 5,
    k: int = 60,
) -> List[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    fused: Dict[str, RetrievalResult] = {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = dict(result)
                fused[chunk_id]["rrf_score"] = 0.0
                fused[chunk_id]["retrieval_source"] = "hybrid"
            fused[chunk_id]["rrf_score"] = float(fused[chunk_id].get("rrf_score", 0.0)) + 1 / (k + rank)
            if result.get("retrieval_source") == "vector":
                fused[chunk_id]["vector_score"] = float(result.get("score", 0.0))
            if result.get("retrieval_source") == "bm25":
                fused[chunk_id]["bm25_score"] = float(result.get("score", 0.0))

    results = list(fused.values())
    for result in results:
        result["score"] = float(result.get("rrf_score", 0.0))
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


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
        model=model,
    )
    bm25_results = bm25_search_chunks(query=query, chunks=chunks, top_k=top_k)
    return rrf_fuse_results([vector_results, bm25_results], top_k=top_k)


def rerank(
    query: str,
    candidates: List[RetrievalResult],
    top_k: int | None = None,
    rerank_model: str | None = None,
) -> List[RerankedResult]:
    cfg = get_config()
    top_k = top_k if top_k is not None else int(cfg.get("rerank_top_k", 5))
    rerank_model = str(rerank_model or cfg.get("rerank_model", _DEFAULT_RERANK_MODEL))

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not candidates:
        return []
    
    reranker = _get_reranker(rerank_model)

    valid_candidates = [item for item in candidates if item.get("text")]
    pairs = [(query, item["text"]) for item in valid_candidates]
    reranked = []

    scores = reranker.predict(pairs)

    for item, score in zip(valid_candidates, scores, strict=True):
        result = dict(item)
        result["rerank_score"] = float(score)
        result["score"] = float(score)
        result["retrieval_source"] = "rerank"
        reranked.append(result)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    return reranked[:top_k]


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

    candidates = hybrid_search_pdf_chunks(
        query=query,
        chunks=chunks,
        top_k=candidate_k,
        collection_name=collection_name,
        model=model,
    )
    rerank_results = rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
        rerank_model=rerank_model,
    )

    return rerank_results


def build_context(rerank_results: List[RerankedResult], max_chars: int = 6000) -> str:
    blocks = []
    used_chars = 0
    
    for index, item in enumerate(rerank_results, start=1):
        source = item.get("source", "")
        page_start = item.get("page_start", "")
        page_end = item.get("page_end", "")
        heading = item.get("heading", "")
        text = item.get("text", "").strip()
        
        if not text:
            continue

        block = (
            f"[{index}]\n"
            f"Source: {source}\n"
            f"Pages: {page_start}-{page_end}\n"
            f"heading: {heading}\n"
            f"text: {text}\n"
        )
        next_size = len(block) + 2
        if used_chars + next_size > max_chars:
            break

        blocks.append(block)
        used_chars += next_size
        
    return "\n\n".join(blocks)


def build_prompt(context: str, query: str) -> str:
    # 1. 系统/角色说明
    instruction = """
你是一个论文回答助手。
你只能基于给定 Context 回答问题。
如果 Context 中没有足够证据，就明确说“根据当前论文内容无法回答”。
回答时必须使用[1]、[2]这样的引用编号。
不要编造论文中没有出现的信息。
""".strip()
    
    # 2. 用户问题
    question_block = f"""
Question:
{query}
""".strip()
    
    # 3. 检索上下文
    context_block = f"""
Context:
{context}
""".strip()
    
    # 4. 输出要求
    output_format = """
Answer:
请给出简洁、准确的回答，并在关键结论后附上引用编号。
""".strip()
    
    # 5. 拼成最终 prompt
    prompt = "\n\n".join([
        instruction,
        question_block,
        context_block,
        output_format,
    ])

    return prompt

def call_llm(prompt: str) -> str:
    cfg = get_config()

    base_url = str(cfg.get("llm_url", "https://api.deepseek.com"))
    model = str(cfg.get("llm_model", "deepseek-chat"))
    llm_api_key_env = str(cfg.get("llm_api_key_env", "DEEPSEEK_API_KEY"))

    api_key = os.getenv(llm_api_key_env, "")
    if not api_key:
        raise ValueError(f"{llm_api_key_env} must be set.")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt,}],
    )

    return response.output_text

def answer(
    query: str,
    candidate_k: int | None = None,
    embed_model: str | None = None,
    rerank_model: str | None = None,
    top_k: int | None = None,
    collection_name: str | None = None,
    max_context_chars: int | None = None,
) -> AnswerResult:
    cfg = get_config()
    candidate_k = candidate_k if candidate_k is not None else int(cfg.get("rerank_candidate_k", 20))
    top_k = top_k if top_k is not None else int(cfg.get("rerank_top_k", 5))
    collection_name = str(collection_name or cfg.get("collection_name"))
    embed_model = str(embed_model or cfg.get("embed_model"))
    rerank_model = str(rerank_model or cfg.get("rerank_model"))
    max_context_chars = max_context_chars if max_context_chars is not None else int(cfg.get("max_context_chars", 6000))

    rerank_results = retrieve_with_rerank(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        collection_name=collection_name,
        model=embed_model,
        rerank_model=rerank_model,
    )
    context = build_context(rerank_results, max_chars=max_context_chars)
    prompt = build_prompt(context, query)
    answer_text = call_llm(prompt)

    sources: List[AnswerSource] = [
        {
            "source": item["source"],
            "chunk_index": item["chunk_index"],
            "page_start": item["page_start"],
            "page_end": item["page_end"],
            "heading": item["heading"],
            "score": item["score"],
            "retrieval_source": item["retrieval_source"],
        }
        for item in rerank_results
    ]
    return {
        "query": query,
        "answer": answer_text,
        "sources": sources,
    }
