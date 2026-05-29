from sentence_transformers import SentenceTransformer       # Embedding model
from typing import List

def is_heading_title(sente: str) -> bool:
    return

def fixed_size_chunks(file_path: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [text[i:i+chunk_size] for i in range(0, len(text), step)]

