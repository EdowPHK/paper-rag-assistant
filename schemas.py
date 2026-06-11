from typing import List, TypedDict

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