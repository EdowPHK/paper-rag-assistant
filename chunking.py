from typing import List
from sentence_transformers import SentenceTransformer
import re
import hashlib

from schemas import PdfPage, PdfChunk
from config import get_config
from embeddings import get_encoder

def _make_chunk_id(source: str, page_start: int, page_end: int, heading: str, text: str) -> str:
    raw = f"{source}:{page_start}:{page_end}:{heading}:{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


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
    tokenizer = get_encoder(model).tokenizer

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