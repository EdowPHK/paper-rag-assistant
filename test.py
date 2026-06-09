import unittest
import sys
import types
from unittest.mock import patch


sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

qdrant_client = types.ModuleType("qdrant_client")
qdrant_models = types.ModuleType("qdrant_client.models")


class FakeDistance:
    DOT = "Dot"


class FakeVectorParams:
    def __init__(self, size: int, distance: str) -> None:
        self.size = size
        self.distance = distance


class FakePointStruct:
    def __init__(self, id: str, vector: list[float], payload: dict) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


qdrant_models.Distance = FakeDistance
qdrant_models.VectorParams = FakeVectorParams
qdrant_models.PointStruct = FakePointStruct
qdrant_client.QdrantClient = object
qdrant_client.models = qdrant_models
sys.modules.setdefault("qdrant_client", qdrant_client)
sys.modules.setdefault("qdrant_client.models", qdrant_models)

pymupdf = types.ModuleType("pymupdf")
sys.modules.setdefault("pymupdf", pymupdf)

import demo


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class FakeEncoder:
    tokenizer = FakeTokenizer()


class ChunkingTests(unittest.TestCase):
    def test_split_pdf_pages_into_chunks_uses_heading_and_token_window(self) -> None:
        pages = [
            {
                "source": "paper.pdf",
                "page_id": 1,
                "text": "1 Introduction\nalpha\nbeta\ngamma",
            },
            {
                "source": "paper.pdf",
                "page_id": 2,
                "text": "delta\nepsilon",
            },
        ]

        with patch.object(demo, "_get_encoder", return_value=FakeEncoder()):
            chunks = demo.split_pdf_pages_into_chunks(
                pages,
                target_tokens=12,
                overlap_tokens=2,
                model="fake-model",
            )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["source"] == "paper.pdf" for chunk in chunks))
        self.assertTrue(all(chunk["heading"] == "1 Introduction" for chunk in chunks))
        self.assertTrue(all(len(FakeTokenizer().encode(chunk["text"])) <= 12 for chunk in chunks))
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(isinstance(chunk["chunk_id"], str) for chunk in chunks))

    def test_split_pdf_pages_into_chunks_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            demo.split_pdf_pages_into_chunks([], target_tokens=10, overlap_tokens=10, model="fake-model")


if __name__ == "__main__":
    unittest.main()
